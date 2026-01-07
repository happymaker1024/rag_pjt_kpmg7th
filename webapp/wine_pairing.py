from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


from dotenv import load_dotenv
import os

load_dotenv(override=True, dotenv_path="../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# LLM을 통한 요리 정보 설명
# 1. 함수 정의 : 이미지 -> 요리명, 풍미 설명 출력
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

# 2. 함수 정의 : 요리 설명 -> 요리 설명, 와인 추천(Top-5)
# 요리에 어울리는 와인 top-5 검색결과를 리턴하는 함수 정의
def search_wines(query):
    embedding = OpenAIEmbeddings(
         model = OPENAI_EMBEDDING_MODEL
    )
    
    # 벡터 db에서 유사도계산, top-5 검색
    # 벡터 db 객체 생성
    vector_db = PineconeVectorStore(
        embedding = embedding,  # 질문에 대한 임베딩 벡터가 생성됨
        index_name = PINECONE_INDEX_NAME ,
        namespace = PINECONE_NAMESPACE
    )
    # 벡터 db에서 질문과 가장 유사한, top-5 검색하기
    results = vector_db.similarity_search(query, k=5)  # top-5 검색

    context = "\n".join([doc.page_content for doc in results])    

    # 함수를 호출한 쪽으로 query, top-5의 검색 결과에 필터링한 결과를 리턴함
    return {
        "query": query,
        "wine_reviews": context
    }

# 함수3 (r3). 요리설명, top-5의 context 입력 받고 -> 요리에 어울리는 와인 추천
def recommend(input_data):
    prompt = ChatPromptTemplate([
    ("system", """
    🍷 Wine Sommelier – System Prompt (Short / Optimized)
    You are a professional wine sommelier specialized in food and wine pairing.
    When responding, you:
    - Analyze food characteristics (ingredients, cooking method, sauce, flavor intensity)
    - Consider wine structure (acidity, tannin, sweetness, body, alcohol)
    - Apply pairing logic (balance, contrast, complement, intensity matching)
    You always:
    - Explain why a pairing works
    - Adapt recommendations to the customer’s taste, budget, and occasion
    - Use clear, accessible language and avoid unnecessary jargon
    Your goal:
    Recommend wine pairings that create harmony between food and wine and maximize the customer’s enjoyment.
        """),
        ("human", """ 아래의 와인리뷰 내용에서만 추천을 해줘 
        요리 설명 : {query}
        와인 리뷰 : {wine_reviews}

        답변은 json으로 다음과 같이 응답해 주세요.
        recommend_wine:
        recommend_reason:
        """)
    ])

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )
    # str 파서
    # output_parser = StrOutputParser()

    # json 파서
    output_parser = JsonOutputParser()

    # pipeline : 데이터의 흐름
    chain = prompt | llm | output_parser

    return chain

# 함수를 실행하기
def wine_pair_main(img_url):
    # RunnableLambda 객체 생성(데이터 파이프라인을 연결하기 위해)
    r1 = RunnableLambda(describe_dish_flavor)
    r2 = RunnableLambda(search_wines)
    r3 = RunnableLambda(recommend)

    # chain으로 연결하기
    chain = r1 | r2 | r3

    # RunnableLambda를 통한 함수 실행
    input_data = {
        "image_url": img_url
    }

    res = chain.invoke(input_data)
    # print(res)
    return res

# 모듈 테스트용 코드
if __name__ == "__main__":
    print(__name__)
    print("-"*30)
    img_url = "https://thumbnail.coupangcdn.com/thumbnails/remote/492x492ex/image/vendor_inventory/9d0d/fd3f0d77757f64b2eba0905dcdd85051932ec1ab5e6afc0c3246f403fabc.jpg"
    result = wine_pair_main(img_url)
    print(result)