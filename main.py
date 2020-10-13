# main program of VVSec

from adv_attack import *


def test_dataset(ref, user):
    file1 = ref  # reference input depth file name
    inp1 = create_input_rgbd(file1)
    file2 = user  # user input depth file name
    inp2 = create_input_rgbd(file2)

    inp2_adv, similarity, time_cost, is_converged = cw_mask(steps=2000, lr=0.1, eps=0.00, strength=0, Img1=inp1,
                                                            Img2=inp2, depth=False, clip=32)

    noise = abs(inp2_adv - inp2)
    original_similarity = model_recover.predict([inp1, inp2])[0][0]
    new_similarity = model_recover.predict([inp1, inp2_adv])[0][0]
    l2_norm = np.linalg.norm(noise) / np.linalg.norm(inp2)

    print("original similarity", original_similarity)
    print("new similarity", new_similarity)
    print("l2 norm", l2_norm)
    print("time cost", time_cost)

    plot_test(inp1, inp2, inp2_adv, original_similarity, new_similarity, l2_norm, time_cost)


if __name__ == '__main__':
    model_recover = get_model()

    # please use the depth file name, which ends in ".dat", as the input

    dataset1_reference_input = 'dataset/ds1/0.jpg_d.dat'
    dataset1_user_input = 'dataset/ds1/1.jpg_d.dat'
    test_dataset(dataset1_reference_input, dataset1_user_input)

    dataset1_reference_input = 'dataset/ds1/0.jpg_d.dat'
    dataset1_user_input = 'dataset/ds1/9.jpg_d.dat'
    test_dataset(dataset1_reference_input, dataset1_user_input)
