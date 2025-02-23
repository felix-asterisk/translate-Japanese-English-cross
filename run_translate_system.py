import sys
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

LLM_MODEL = "gemma2:9b"

DESC_TO_MODEL = """あなたは優秀な言語翻訳者です。
日本語が入力された場合、英語に翻訳してください。
それ以外の言語の場合は日本語に翻訳してください。
翻訳は、可能であれば1つ以上から3つまで翻訳例を出してください。
翻訳が不可能であれば、'翻訳はできませんでした'と回答してください。"""

INIT_SYS_MESSAGE = """日英相互翻訳システムと会話を始めましょう。
本システムでは、下記のルールで翻訳されます。
Japanese -> English
Other -> Japanese
終了するには '/bye' と入力してください。"""

def main():
    # LLMを定義
    chat_model = OllamaLLM(model=LLM_MODEL)

    # メモリを初期化
    memory = ConversationBufferMemory(return_messages=True)

    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(DESC_TO_MODEL),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # 会話チェーンを初期化
    conversation = ConversationChain(
        llm=chat_model,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    print(INIT_SYS_MESSAGE)

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == '/bye':
            print("チャットを終了します。")
            break

        response = conversation.predict(input=user_input)
        print("Translator:", response)

if __name__ == "__main__":
    main()

