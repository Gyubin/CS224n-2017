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

## 3. Neural Machine Translation

Slide 16-28

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

## 4. Attention mechanism

Slide 29-50

- 기존 encoder-decoder 모델에서 하나의 fixed-dimensional 벡터가 전체 문장을 다 표현하기가 부족한 경우가 많다. 긴 문장일 경우에 특히.(slide 29)
- 해결책은 source의 hidden state들을 RAM처럼 취급해서 언제든지 필요할 때 끌어오는 방식. 가장 기본적인 개념은 vision 분야에서 처음 나왔다.(slide 30)
- 이를 source와 translation의 "implicit alignment"라고도 표현. 사람 번역가도 이런 식으로 한다.(slide 30)
- SMT에선 이런 작업을 특정 모델이 담당한다. EM 알고리즘으로.(slide 31) 그리고 NMT는 attention이 담당하는 것이고 시각화하면 재밌게 그려진다(slide 32)

![attention-score](https://i.imgur.com/2ScIXIV.png)

- Score: attention이 어느 부분을 중점적으로 보는지의 측정 기준(slide 33-43)
- 각 hidden state의 score : 해당 state(`h_s`)와, 최종적으로 endcoded된 hidden state(`h_{t-1}`)의 dot product로 계산된다. 위 이미지처럼 스칼라 값.
    + dot product를 할 때 중간에 W 매트릭스가 들어간다.
    + `(h_t).T`, `W`, `h_s` 가 순서대로 dot product 되는 값이 score
    + W 매트릭스가 학습되면서 가중치가 학습되는 것
- 계산된 score를 softmax로 normalize.
- 각 hidden state에 가중치를 곱해서 element wise summation 한다. 그렇게 나온 것이 Context vector다.
- Context vector, final encoded hidden state, input이 혼합되어 최종 `h_t` 벡터가 된다.

![global-local](https://i.imgur.com/8tvbdu4.png)

- 가중치를 계산하는 베이스를 모든 state로 둘 것인지, 부분만 볼 것인지에 따라 global, local이 나뉜다.
- local 방식이 long sequence에서 더 나은 성능을 보인다. 다 볼 것인가, 부분만 선택해서 가중
- fertility(slide 50)
    + coverage의 반대말
    + 너무 특정 hidden state만 많이 attention하면 나쁘다.
    + 1 word를 6 word로 번역한다면 반복 사용되는 단어가 많을 것. 안좋다. 피하자.

## 5. Ways of Decoding

Slide 51-58

- Exhaustive search: 모든거 다 계산한다 절대 하면 안된다.
- Ancestral sampling: 조건부 확률을 사용한 가장 이상적이고 이론적인 방식. 하지만 자연어가 아주 reasonable한 형태로(주어, 서술어, 목적어 딱딱 있고, 반복단어 없는) 있을 경우가 드물고기 때문에 variance가 높다고 한다.
- Greedy search: 다음 단어를 뽑을 때 그 순간 최고의 값을 지니는 단어만 계산하고 그에 대해서 가지를 뻗어나간다. 매우 효율적이고 빠르지만 심각한 suboptimal 문제
- Beam search: 가장 많이 사용된다.
    + Greedy가 순간 순간 최고 스코어 하나만 뽑았다면 여기선 여러개를 뽑는다. 뽑는 후보를 beam이라고 하며 5개, 10개가 de facto다.
    + 처음 5개를 뽑았으면 그 5개와 연결해서 또 각각 5개를 뽑는다. 이 때의 25개의 결과 중 최고의 확률값을 가지는 5개를 다시 뽑고, 또 각각 5개를 뽑고 추리는 것을 반복한다.
    + beam을 무한개로 하면 exhaustive search 형태가 되므로 지양하고, cross validation을 통해 최적의 k개를 찾아내면 좋다.
