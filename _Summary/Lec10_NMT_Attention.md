# Lec 10. NMT and Models with Attention

## 1. Intro

Slide 1-5

- 번역 서비스는 대부분 무료 서비스로 존재한다.
- 유럽이나 아시아에서 다양한 분야에서 중요한 이슈(social, government, military etc)
- 미국 토박이들은 영어로 대부분 가능해서 중요성을 못 느끼기도 한다.
- NMT는 슬로건같은 것이고, "Neural + Machine Translation"이 진짜 의미다.

## 2. Encoder-Decoder Architecture

Slide 6-

- 기본적인 아키텍처(slide 6, 9), 층을 더 깊게 쌓은 Deep RNN(slide 10)
- 2012년 딥러닝이 주목받기 전에도 여러 모델이 있었다. 하지만 컴퓨팅 파워, 데이터 등의 리소스 문제로 좋은 성능을 내진 못했음(slide 7, 8)
- GRU나 LSTM에서 forget gate는 거의 잊지 않는 형태로 작용한다. 모든 문장을 다 잘 기억해야 내용 빠짐없이 번역 가능. 그래서 LSTM이 좋은 효과를 낸다. Vanila RNN은 성능 안나옴.
- Machine Translation 처음엔 GRU 쓰지 않고, "Recurrent sequence of
convolutional networks"를 썼다고 한다. by 딥마인드 연구자들
- 조경현 교수 팀이 한 방식 중 every time step에 encoded 결과물 Y가 feed되는 아키텍처가 있다. 이게 좀 더 발전되어 attention과 연결된다.(slide 13)
