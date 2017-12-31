# Lec 10. NMT and Models with Attention

## 1. Intro

Slide 1-5

- 번역 서비스는 대부분 무료 서비스로 존재한다.
- 유럽이나 아시아에서 다양한 분야에서 중요한 이슈(social, government, military etc)
- 미국 토박이들은 영어로 대부분 가능해서 중요성을 못 느끼기도 한다.
- NMT는 슬로건같은 것이고, "Neural + Machine Translation"이 진짜 의미다.

## 2. Encoder-Decoder Architecture

Slide 6-13

- 기본적인 아키텍처(slide 6, 9), 층을 더 깊게 쌓은 Deep RNN(slide 10)
- 2012년 딥러닝이 주목받기 전에도 여러 모델이 있었다. 하지만 컴퓨팅 파워, 데이터 등의 리소스 문제로 좋은 성능을 내진 못했음(slide 7, 8)
- GRU나 LSTM에서 forget gate는 거의 잊지 않는 형태로 작용한다. 모든 문장을 다 잘 기억해야 내용 빠짐없이 번역 가능. 그래서 LSTM이 좋은 효과를 낸다. Vanila RNN은 성능 안나옴.
- Machine Translation 처음엔 GRU 쓰지 않고, "Recurrent sequence of
convolutional networks"를 썼다고 한다. by 딥마인드 연구자들
- 조경현 교수 팀이 한 방식 중 decoder의 every time step에 encoded 결과물 Y가 feed되는 아키텍처가 있다. 이게 좀 더 발전되어 attention과 연결된다.(slide 13)

### 3. Neural Machine Translation

- Four big wins of NMT(slide 16)
    + End to end training: 하나의 큰 objective function을 가지고 전체 네트워크가 학습을 한다는 측면에서 큰 의미를 지니고, 생산적이다.
    + Distributed representation, Better exploitation of context : 4-5 gram이 큰 context를 보기 떄문에 의미가 있지만 이 정도 범위로는 부족하다. 그리고 기존 statistical machine translation에선 4-5 gram 이상을 사용하면 sparsity 문제로 엉망이 된다. NMT는 더 큰 범위를 보면서 좋은 성능을 냄
    + More fluent text generation: 번역 질은 안 좋을 수 있는데 자연스러운 문장은 매우 잘 생성한다.
- 추가로 특성들(slide 17)
    + SMT에선 이런저런 모델들을 파이프라인으로 연결해서 태스크 수행했다. 각각의 모델들마다 역할이 명확하게 정해져있음. NMT는 end-to-end 방식으로 Black box model
    + SMT와 다르게 NMT에선 syntactic, semantic 구조를 명시적으로 사용하지 않는다.
    + NMT에선 부정의 부정이나 대명사 사용 등의 것들이 잘 해결되지 않는다.
- NMT의 큰 셀링 포인트 중 하나는 아주 compact 하다는 것. 오프라인으로 스마트폰에서 구동 가능하다. 여행 갔을 때 편리하게 사용 가능(slide 19)

![google-nmt](http://tsong.me/public/img/reading/google-nmt-lstm.png)

- Google's multilingual NMT system
    + 처음에는 모든 language pair에 대해서 각각 학습하고 시스템을 만들었다. 구글이 다루는 언어가 약 80여가지라서 6400개를 만드는 셈. 결국 기각
    + 다음엔 "shared encoder + 언어별 decoder" 혹은 "언어별 encoder - shared decoder"를 쓰는 형태. 역시 대체되었다.
    + 최종 shared system(encoder-decoeder)을 쓰는 형태로 확정
- Google NMT system 장점
    + 완전한 싱글 모델
    + 마치 시스템 자신만의 interlingua를 만들어서 그걸 사이에 두고 번역을 수행. SMT보다 NMT가 이런 interlingua 적인 접근이 더 쉽다고 한다.
    + 그래서 데이터 적은 언어도 성능이 향상되고, 트레이닝 때 보지 못한 데이터도 번역 가능(zero-shot translation)
- 언어 구분은 input source 맨 앞에 인위적인 token을 집어넣는다. `2es`라면 english to spanish를 의미. 이렇게 구분해서 하나의 시스템으로 번역 가능
