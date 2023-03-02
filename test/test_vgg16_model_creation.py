import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test():
    from hololive_categorizer import vgg16_functinal_model, vgg16_model

    model_func = vgg16_functinal_model()
    model_func.summary()

    model_seq = vgg16_model()
    model_seq.summary()


if __name__ == "__main__":
    test()
