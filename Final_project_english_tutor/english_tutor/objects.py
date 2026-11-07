"""아이에게 알려줄 사물 사전.

YOLO(COCO 80클래스)가 탐지하는 이름 -> (영어 단어, 한국어 이름)
- 영어 단어는 LLM에게 맡기지 않고 여기서 확정합니다. (작은 모델의 오답/환각 방지)
- Gemma는 이 단어를 받아 '쉬운 예문과 설명'만 생성합니다.
- 칼, 가위처럼 아이에게 위험한 물건은 일부러 뺐습니다.
"""

KID_OBJECTS = {
    "book": ("book", "책"),
    "cup": ("cup", "컵"),
    "teddy bear": ("teddy bear", "곰 인형"),
    "bottle": ("bottle", "물병"),
    "sports ball": ("ball", "공"),
    "banana": ("banana", "바나나"),
    "apple": ("apple", "사과"),
    "orange": ("orange", "오렌지"),
    "chair": ("chair", "의자"),
    "clock": ("clock", "시계"),
    "cell phone": ("phone", "전화기"),
    "remote": ("remote control", "리모컨"),
    "keyboard": ("keyboard", "키보드"),
    "mouse": ("mouse", "마우스"),
    "laptop": ("laptop", "노트북"),
    "bowl": ("bowl", "그릇"),
    "spoon": ("spoon", "숟가락"),
    "fork": ("fork", "포크"),
    "backpack": ("bag", "가방"),
    "umbrella": ("umbrella", "우산"),
    "kite": ("kite", "연"),
    "frisbee": ("frisbee", "원반"),
    "vase": ("vase", "꽃병"),
    "potted plant": ("plant", "화분"),
    "toothbrush": ("toothbrush", "칫솔"),
    "tv": ("TV", "텔레비전"),
    "couch": ("sofa", "소파"),
    "bed": ("bed", "침대"),
    "cat": ("cat", "고양이"),
    "dog": ("dog", "강아지"),
    "bird": ("bird", "새"),
}
