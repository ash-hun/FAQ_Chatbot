# FAQ_Chatbot
네이버 스마트스토어의 자주 묻는 질문(FAQ)을 기반으로 질의응답하는 챗봇 만들기

---  


### 개발환경구성
- [x] 환경구성

### 데이터 전처리
- [x] pickle 데이터를 csv로 정제

### 베이스라인 설계
- [x] BaseLine 개발

### 벡터 데이터베이스 활용
- [x] ChromaDB 생성

### 기능구현
- RAG pipeline 개발
  - [x] pipeline 개발
  - [x] 대화기록 저장 및 맥락 반영하여 답변 생성

### 성능향상
- [x] 성능 최적화 (비용, 시간대비 품징향상, 기타 등등)
  - 고민한 사항
    - 데이터 전처리
      - 의미없는 공통부분 삭제
    - Chunksize 결정
      - 전체 item의 mean length 선정
    - Embedding Model 선정
      - 이전 실험 토대로 한국어 임베딩 모델 선정
    - 관련없는 내용 예외처리 방식
      - `prompt engineering` vs `after response process`
    - 메모리 구성
      - Chat history include type : `Raw Message` vs `Summerizing`


### 평가
- [x] 질의응답 시나리오 구성 및 정성평가
- [x] 답변품질 확인

### 정리
- [ ] 코드 정리 및 배포환경 구성
- [ ] 노션 정리

### 최종 결과확인
![alt text](./assets/demo_image.png)