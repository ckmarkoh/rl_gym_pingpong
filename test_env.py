
from env_pingpong_dis import PingPongEnvDis

def test1():
    p = PingPongEnvDis()
    ball = [4,2,1,1]
    bar = [3,0]
    state = p.encode(ball,bar)
    ball2,bar2 = p.decode(state)
    assert ball == ball2
    assert bar == bar2
    print("test1 pass")

def test2():
    p = PingPongEnvDis()
    for h in range(p.b_height):
        for w in range(p.b_width):
            for d1 in range(2):
                for d2 in range(2):
                    for br in range(p.b_height):
                        ball = [h, w, d1, d2]
                        bar = [br, 0]
                        state = p.encode(ball, bar)
                        ball2, bar2 = p.decode(state)
                        assert ball == ball2
                        assert bar == bar2
    print("test2 pass")


if __name__ == '__main__':
    test1()
    test2()
