name = "철수"
age = 20
height = 178.31754081

print(f"이름: {type(name)}, 나이: {type(age)}, 키: {type(height)}")
print()
print("이름:", name, end=", ")
print("나이: " + str(age) + "세", end=", ")
print(f"키(m): {height/100:.2f}m", end=", ")
print(f"키(cm):", str(height)[0:5] + "cm", end="\n\n")
print("성인 여부:", age > 19)

score_str = "0.8825"

# TODO: score_str를 백분율로 표현한 뒤 정수로 변환하여 출력
score_int = int(float(score_str)*100)
print(f"최종 점수: {score_int}점") 

scores = [[45, 88, 92],
          [30, 75, 60],
          [95, 40, 50]]

# TODO: List Comprehension을 활용해 50점 이상인 값만 통과시켜 담으세요.
passed_scores = [score for row in scores for score in row if score>=50]
print("합격 점수들:", passed_scores)

students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]

# TODO: enumerate와 zip을 조합하거나 활용하여 80점 이상 학생 정보 출력
# TODO: 예상 출력 "1번 짱구 학생 (점수: 85점)"
for idx, (name, score) in enumerate(zip(students, scores), start=1):
    if score>=80:
        print(f"{idx}번 {name} 학생 (점수: {score}점)")


import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np_arr.sum())
print("Minimum:", np_arr.min())
print("Maximum:", np_arr.max())
print("Mean:", np_arr.mean())
print("Argmax:", np_arr.argmax())

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = (np_arr - np.min(np_arr)) / (np.max(np_arr) - np.min(np_arr))
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = np.array(np_arr_normalized, dtype=np.float32)
print(np_arr_final)
print(np_arr_final.dtype)

raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]

# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [s.split("_")[1] for s in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)


# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 15개 성적 중 85점 이상 개수
over_85 = int((raw_scores >= 85).sum())
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = raw_scores.mean(axis=1)
mean_str_list = [f"{student_names[i]} {student_means[i]:.0f}점" for i in range(len(student_names))]
print("4. 학생별 평균 점수: " + ", ".join(mean_str_list))

# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = int(raw_scores.mean(axis=1).argmax())
top_student_name = student_names[top_student_idx]
top_student_mean = student_means[top_student_idx]
print(f"5. 최고 평균 수석 학생: {top_student_name} ({top_student_mean:.2f}점)")

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = int(raw_scores.max())
overall_max_idx = int(raw_scores.argmax())
num_subjects = len(subjects)
overall_max_subject = subjects[overall_max_idx % num_subjects]
overall_max_student = student_names[overall_max_idx // num_subjects]
print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")