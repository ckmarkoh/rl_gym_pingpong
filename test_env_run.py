
from env_pingpong import PingPongEnv


def test1():

    enviroment = PingPongEnv()
    for _ in range(10) :
        print("--------------new game--------------")
        state = enviroment.reset()
        enviroment.render()
        action = 1
        while True:
            if   enviroment.bar[0] > 1:
                action = 0
            elif enviroment.bar[0] < 1:
                action = 2
            else:
                action =1
            state, reward, terminated, info = enviroment.step(action)
            print("reard:%s"%reward)
            enviroment.render()
            if terminated:
                print("terminated")
                break
    pass



if __name__ == '__main__':
    test1()