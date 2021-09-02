import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
# 시그모이드 함수 , 활성화 함수  - activation function
# 1  - (b) \
# x1 - (W1) - - - - (a) - - - (y)  # 행렬을 이용해서 가중치와 입력값을 연산한다.
# x2 - (W2)/
def init_network():
    network = {} # {} = dictionary  , dictionary 를 선언, 초기화 할때 사용함
    # [] 는 key 에 해당하는 value를 할당할 떼 사용
    network['W1'] = np.array([[0.1, 0.3 , 0.5] , [0.2 ,0.4, 0.6]]) #퍼셉트론 1차 가중치
    network['b1'] = np.array([0.1, 0.2 ,0.3]) # # 1차 편향값
    network['W2'] = np.array([[0.1,0.4],[0.2, 0.5],[0.3,0.6]]) # 퍼셉트론 2차 가중치
    network['b2'] = np.array([0.1,0.2]) # 2차 편향값
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]]) # 퍼셉트론 3차 가중치
    network['b3'] = np.array([0.1,0.2]) # 3차 편향값

    return network

def forward(network,x): # 입력신호를 함수로 처리하는 과정
    W1 , W2, W3 = network['W1'] , network['W2'] ,network['W3']
    b1,  b2, b3 = network['b1'] , network['b2'] ,network['b3']

    a1 = np.dot(x ,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    Y = a3

    return Y

network = init_network() # 함수 불러오기
x = np.array([1.0 , 0.5 ]) # 행렬 연산
y = forward(network, x)
print(y)







