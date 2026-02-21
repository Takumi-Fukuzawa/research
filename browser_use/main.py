import os
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserProfile, ChatGoogle
import asyncio

from situation_summarizer import summarize_page_state_from_session

async def main():
    load_dotenv()

    # APIキーチェック
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is not set')
    
    print('-' * 30)
    print('browser-use エージェントシステムを起動します')
    print('-' * 30)

    # 1. LLMの初期化
    llm = ChatGoogle(model='gemini-2.5-flash', api_key=api_key)

    # 2. ブラウザの起動
    browser = Browser(
        browser_profile=BrowserProfile(
            headless=False,
            no_viewport=True,
            keep_alive=True,
        )
    )

    # 【連携の肝】Agentと要約機能で「同じ画面」を共有するためにコンテキストを作成
    # context = await browser.new_context()

    # 【重要】ライブラリが自動設定する固定ウィンドウサイズを解除する
    # これにより --window-size 引数が消え、代わりに --start-maximized が有効になります
    if hasattr(browser, 'browser_profile'): # 安全のため属性チェック追加
        browser.browser_profile.window_size = None
        # 【追加】位置固定も解除する（これをしないと最大化されないことがあります）
        browser.browser_profile.window_position = None

    # メインループ
    try:
        while True:
            print("\n" + "="*40)
            print("[メニュー]")
            print("1: タスク実行 (Agent)")
            print("2: 状況要約 (Summarizer)")
            print("q: 終了")
            print("="*40)

            choice = input("選択してください >> ").strip()

            if choice == 'q':
                print("終了します。")
                break

            elif choice == '1':
                # --- タスク実行 ---
                task_content = input('実行したいタスクを入力してください >> ')
                if not task_content: continue

                # Agent初期化（ご提示のコードを維持）
                agent = Agent(
                    task=task_content,
                    llm=llm,
                    browser=browser, # ★修正: contextを渡す
                    flash_mode=True,
                )

                print(f'\n>>> Agentを実行します: {task_content}')
                try:
                    await agent.run()
                except Exception as e:
                    print(f"Agent実行中にエラーが発生しました: {e}")

            elif choice == '2':
                # --- 状況要約 ---
                user_query = input('知りたいこと（空なら「今の画面はどうなっている？」）>> ').strip()
                if not user_query:
                    user_query = '今の画面はどうなっている？'

                print(f'\n>>> 状況を要約します...')
                try:
                    # エージェント実行後の「今の画面」の状況要約
                    summary = await summarize_page_state_from_session(
                        user_query=user_query,
                        session=browser,  # ★修正: Browserではなくcontextを渡す
                        llm=llm,
                    )
                    print("\n" + "-" * 30)
                    print("現在の画面の状況要約:")
                    print("-" * 30)
                    print(summary)
                except Exception as e:
                    print(f"要約中にエラーが発生しました: {e}")
                    print("※まだブラウザが開いていない場合、先に「1」で何らかのタスク（例: 'Googleを開いて'）を実行してください。")

    finally:
        # 終了処理
        print("ブラウザを閉じています...")
        
        try:
            if hasattr(browser, 'kill'):
                await browser.kill()
            elif hasattr(browser, 'stop'):
                await browser.stop()
            else:
                print("※ 終了メソッドが見つかりませんでした。")

            print("システムを正常に終了しました。")
        except Exception as e:
            print(f"ブラウザの終了中にエラーが発生しました: {e}")


if __name__ == '__main__':
    asyncio.run(main())