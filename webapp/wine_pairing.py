from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
import os

load_dotenv(override=True, dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM을 통한 요리 정보 설명
# 함수 정의 : 이미지 -> 요리명, 풍미 설명 출력
def describe_dish_flavor(input_data):

    prompt = ChatPromptTemplate([
        ("system", """
        You are a culinary expert who analyzes food images.
        When a user provides an image of a dish,
        identify the commonly recognized name of the dish, and
        clearly and concisely describe its flavor, focusing on the cooking method, texture, aroma, and balance of taste.
        If there is any uncertainty, base your analysis on the most likely dish, avoid definitive claims, and maintain a professional, expert tone.
        """),
        HumanMessagePromptTemplate.from_template([
            {"text": """아래의 이미지의 요리에 대한 요리명과 요리의 풍미를 설명해 주세요.
            출력형태 :
            요리명:
            요리의 풍미:
            """},
            {"image_url": "{image_url}"} # image_url는 정해줘 있음.        
        ])
    ])
    
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=GOOGLE_API_KEY
    )
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser

    return chain

# 함수를 실행하기
def wine_pair_main(img_url):
    # 함수를 전달인자로 넣기
    r1 = RunnableLambda(describe_dish_flavor)

    # RunnableLambda를 통한 함수 실행
    input_data = {
        "image_url": img_url
    }

    res = r1.invoke(input_data)
    # print(res)
    return res

# 모듈 테스트용 코드
if __name__ == "__main__":
    img_url = "https://thumbnail.coupangcdn.com/thumbnails/remote/492x492ex/image/vendor_inventory/9d0d/fd3f0d77757f64b2eba0905dcdd85051932ec1ab5e6afc0c3246f403fabc.jpg"
    result = wine_pair_main(img_url)
    print(result)