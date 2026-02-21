import logging
import traceback
from dataclasses import dataclass
from typing import Optional

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)

from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary

@dataclass
class SituationSummary:
    raw_response: str
    summary: str
    suggested_actions: str

class SituationSummarizer:
    def __init__(self, browser_context: BrowserSession, llm: BaseChatModel, user_query: str = ""):
        self.browser_context = browser_context
        self.llm = llm
        self.user_query = user_query
        self.logger = logging.getLogger(__name__)

    async def summarize(self) -> SituationSummary:
        try:
            self.logger.info("🔍 [1/4] 画面の状況要約を開始します...")

            # 1. 状態の取得
            state: BrowserStateSummary = await self.browser_context.get_browser_state_summary(include_screenshot=True)
            
            self.logger.info("🔍 [2/4] UIリストのテキスト化を実行中...")
            # 2. 文字列化
            if state.dom_state and hasattr(state.dom_state, 'llm_representation'):
                ui_elements_text = state.dom_state.llm_representation()
            else:
                ui_elements_text = "UI要素リストを取得できませんでした。"

            query_instruction = ""
            if self.user_query and self.user_query != '今の画面はどうなっている？':
                query_instruction = f"\n\nユーザーからは「{self.user_query}」という関心・疑問が提示されています。これに対する回答を含めて要約してください。"

            system_prompt = f"""あなたは優れたUI/UXアナリストです。
現在ユーザーが開いているWebページのスクリーンショットと、検出されたUI要素のリストが提供されます。
これらを空間的に完全に結びつけて分析し、以下の2点を出力してください。{query_instruction}

1. **状況要約**: 
   画面上に映っている画面が現在どういう状況にあるか（何のためのページか、どういう状態か）を簡潔に説明してください。
   
2. **主な操作内容**: 
   このサイト上で可能な操作内容の上位3〜5件程度をリストアップしてください。
   各操作には、必ず対応するUI要素の番号（例: [12]）を併記してください。
"""
            
            self.logger.info("🔍 [3/4] LLMプロンプトを構築中...")
            
            # 変更点3: 公式のクラスを使ってコンテンツリストを構築する
            content_list = []
            
            # テキストの追加
            content_list.append(
                ContentPartTextParam(
                    text=f"以下のUI要素リストと添付のスクリーンショットをもとに、画面状況を要約してください。\n\n## UI要素リスト\n{ui_elements_text}"
                )
            )

            # 画像の追加（スクリーンショットが存在する場合のみ）
            if state.screenshot:
                content_list.append(
                    ContentPartImageParam(
                        image_url=ImageURL(
                            url=f"data:image/png;base64,{state.screenshot}",
                            media_type="image/png"
                        )
                    )
                )
            else:
                self.logger.warning("⚠️ スクリーンショットが取得できませんでした。")

            # 変更点4: 公式の UserMessage と SystemMessage でラップする
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=content_list)
            ]

            self.logger.info("🔍 [4/4] LLMへリクエストを送信中...")
            
            # LLMへの送信
            response = await self.llm.ainvoke(messages)
            
            # 変更点5: browser-useのLLMラッパーの仕様に合わせ、.contentではなく .completion を参照する
            content = getattr(response, 'completion', str(response))
            if not isinstance(content, str):
                content = str(content)
            
            self.logger.info("✅ 状況要約が完了しました")
            
            return SituationSummary(
                raw_response=content,
                summary="詳細は raw_response を参照", 
                suggested_actions="詳細は raw_response を参照"
            )
            
        except Exception as e:
            self.logger.error(f"❌ 要約処理中に致命的なエラーが発生しました:\n{traceback.format_exc()}")
            raise e
    
async def summarize_page_state_from_session(user_query: str, session: BrowserSession, llm: BaseChatModel) -> str:
    """
    main.py から呼び出され、現在のブラウザコンテキストの状態を要約してテキストで返す関数。
    """
    summarizer = SituationSummarizer(browser_context=session, llm=llm, user_query=user_query)
    result = await summarizer.summarize()
    return result.raw_response