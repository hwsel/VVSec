from utils import *
import tensorflow as tf
import datetime


def cw_mask(steps, lr, eps, strength, Img1, Img2, depth, clip, mask_offset=False):
    start_time = datetime.datetime.now()

    depth_mask = np.ones((200, 200, 1)).astype(np.float32)
    if mask_offset:  # add offset if data from dataset 2
        depth_mask = depth_mask * np.max(Img2[0, :, :, 3:])
        depth_mask[:195, 16:] = Img2[0, :, :, 3:][5:, :184]
        depth_mask = np.sign(depth_mask) * (-0.5) + 0.5
    else:
        depth_mask = np.sign(Img2[0, :, :, 3:]) * (-0.5) + 0.5

    with tf.compat.v1.Session() as sess:
        x1_tensor = tf.placeholder(tf.float32, [1, 200, 200, 4])
        x2_tensor = tf.placeholder(tf.float32, [1, 200, 200, 4])
        tf_eps = tf.placeholder(tf.float32)
        if depth:
            delta = tf.Variable(tf.zeros([1, 200, 200, 4], dtype=tf.float32), trainable=True, name='delta')
            adv_img = x2_tensor + tf.clip_by_value(delta, -clip, clip)
            # future work
        else:
            delta = tf.Variable(tf.zeros([200, 200, 3], dtype=tf.float32), trainable=True, name='delta')
            delta = delta * depth_mask
            depth_zero = tf.keras.backend.concatenate([delta, tf.zeros([200, 200, 1])], axis=-1)
            adv_img = x2_tensor + tf.clip_by_value(depth_zero, -clip, clip)
            rgb, depth = tf.split(adv_img, [3, 1], axis=3)
            adv_img = tf.keras.backend.concatenate([tf.clip_by_value(rgb, 0, 255), depth], axis=3)
        model = load_model("./model/model_top.model")
        learning_rate = lr
        optimizer = tf.train.AdamOptimizer(learning_rate)
        y_true = model(x1_tensor)
        prediction = model(adv_img)
        loss_target = K.sqrt(K.sum((K.square(y_true - prediction))))
        loss = tf.math.maximum(0.4 - loss_target, -strength) + tf_eps * tf.nn.l2_loss(delta)
        var = [x for x in tf.global_variables() if 'delta' in x.name]
        training_op = optimizer.minimize(loss, var_list=var)
        init_var = [x for x in tf.global_variables() if '_power' in x.name]
        delta_var = [x for x in tf.global_variables() if 'delta' in x.name]
        sess.run(tf.variables_initializer(init_var))
        sess.run(tf.variables_initializer(delta_var))
        eps = 0
        for i in range(steps):
            _, loss_, loss_t_, adv_img_ = sess.run([training_op, loss, loss_target, adv_img],
                                                   feed_dict={x1_tensor: Img1, tf_eps: eps, x2_tensor: Img2})
            if loss_t_ >= 0.4 + strength:
                sess.close()
                del sess
                return adv_img_, loss_t_, str(datetime.datetime.now() - start_time), True
            print("step #", i, loss_, loss_t_)
    sess.close()
    del sess
    return adv_img_, loss_t_, str(datetime.datetime.now() - start_time), False
