from fastapi import FastAPI
from wine_pairing import wine_pair_main
# FastAPI() 객체 생성
app = FastAPI()

# 홈
# https://thumbnail.coupangcdn.com/thumbnails/remote/492x492ex/image/vendor_inventory/9d0d/fd3f0d77757f64b2eba0905dcdd85051932ec1ab5e6afc0c3246f403fabc.jpg
@app.get("/")
async def home(image_url: str):
    # 사용자 image url을 받음
    print(image_url)
    # 이미지의 요리명, 요리의 풍미 설명(llm) -> wine top-5 검색 -> 요리에 어울리는 와인 추천
    # 1단계 결과 : 이미지의 요리명, 요리의 풍미 설명(llm)
    # img_url = "https://thumbnail.coupangcdn.com/thumbnails/remote/492x492ex/image/vendor_inventory/9d0d/fd3f0d77757f64b2eba0905dcdd85051932ec1ab5e6afc0c3246f403fabc.jpg"

    # result = wine_pair_main(image_url)  # 나중에 다시 활성화하기
    result = "요리명: 스테이크\n\n요리의 풍미:\n이 요리는 고온에서 조리되어 겉은 바삭하게 캐러멜화되고 속은 촉촉하고 부드러운 핑크빛을 띠는 스테이크입니다. 강한 불에서 구워내어 고기 본연의 깊은 풍미와 고소한 육향이 응축되어 있으며, 겉 부분의 갈색 크러스트에서 오는 감칠맛이 특징입니다. 굵은 소금은 육즙의 풍미를 더욱 끌어올리고 짭조름한 맛의 균형을 잡아주며, 통후추는 은은한 매콤함과 향긋함을 더합니다. 신선한 로즈마리 허브는 소나무 숲을 연상시키는 상쾌하고 아로마틱한 향을 더해 전체적인 풍미에 복합성을 부여하며, 육류의 진한 맛과 조화롭게 어우러져 고급스러운 맛의 조화를 이룹니다."

    print(result)
    # res = "llm을 통해 추천 받은 것을 사용자에게 반환"
    return {"message": result}
