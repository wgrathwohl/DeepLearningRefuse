"""
Video Heirarchical Variational Autoencoder Test

Trains an variational autoencoder to take in a video, V = [I_1, I_2, ..., I_T]
and will produce a set of encodings
H = [(s_1, t_1), (s_2, t_2), ..., (s_T, t_T)] where the s and t variables obey the prior:

t_0 ~ N(0, sigma_t0_sqr)
t_i ~ N(s_{i-1}, sigma_t_sqr)

s_i ~ N(s_V, sigma_s_sqr)
s_V ~ N(0, sigma_s0_sqr)

by Will Grathwohl
"""

from vhvae import *
import cv2
import time
from sklearn.manifold import TSNE
import sklearn.svm
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import random

def generate_chair_rotation_dataset(features):
    # features should be [n_vids, n_frames, n_feats]
    X = []
    Y = []
    for vid in features:
        l = len(vid)
        s_ind = random.randint(0, l-5)
        clip = vid[s_ind: s_ind + 5]
        clip = [clip[0], clip[2], clip[4]]
        if random.random() < .5:
            clip.reverse()

        assert(len(clip) == 3)
        # choose +/- 1 label
        if random.random() < .5:
            # pos
            label = 1
        else:
            # neg
            label = 0
            if random.random() < .5:
                clip = [clip[0], clip[2], clip[1]]
            else:
                clip = [clip[1], clip[0], clip[2]]
        clip_feats = [feats[np.newaxis, :] for feats in clip]
        clip_feats = np.concatenate(clip_feats, axis=1)
        X.append(clip_feats)
        Y.append(label)
    X = np.concatenate(X)
    Y = np.array(Y)
    return X, Y


def generate_test_distributions():
    # kinda (totally) hacky
    ls, gs = train(test=True)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    # Run test data through model and collect dataset of S and T features for t-sne
    n_iters = 100
    all_t_feats = []
    all_s_feats = []
    all_s_means = []
    for i in range(n_iters):
        batch = dataset.GET_TEST_BATCH()
        [t_feats, s_feats, vdm] = sess.run(
            [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                videos_placehoder: batch
            }
        )
        #dataset.DisplayData(vdm)

        t_vids = [f for f in t_feats]
        s_vids = [f for f in s_feats]
        s_means = [s_f.mean(axis=0) for s_f in s_vids]

        all_t_feats.extend(t_vids)
        all_s_feats.extend(s_vids)
        all_s_means.extend(s_means)

    t_feats = np.concatenate(all_t_feats)
    s_feats = np.concatenate(all_s_feats)
    shifted_t_feats = np.concatenate(all_t_feats[1:] + [all_t_feats[0]])
    shifted_t_feats_back = np.concatenate([all_t_feats[-1]] + all_t_feats[:-1])
    f = np.concatenate([s_feats, t_feats], axis=1)
    shifted_f = np.concatenate([s_feats, shifted_t_feats], axis=1)
    shifted_back_f = np.concatenate([s_feats, shifted_t_feats_back], axis=1)


    d_name = os.path.join(FLAGS.train_dir, "feature_swap_videos")
    if not os.path.isdir(d_name):
        os.mkdir(d_name)
        os.mkdir(os.path.join(d_name, "original"))
        os.mkdir(os.path.join(d_name, "forward"))
        os.mkdir(os.path.join(d_name, "backward"))
    for i in range(100):
        [vdm] = sess.run(
            [summary_videos], feed_dict={
                frames_z: f[i*16:(i+1)*16]
            }
        )
        [vdm2] = sess.run(
            [summary_videos], feed_dict={
                frames_z: shifted_f[i*16:(i+1)*16]
            }
        )
        [vdm3] = sess.run(
            [summary_videos], feed_dict={
                frames_z: shifted_back_f[i*16:(i+1)*16]
            }
        )
        #dataset.DisplayData(vdm)
        for f_i, im in enumerate(vdm[0]):
            fname = os.path.join(d_name, "original", "vid_{}_frame_{}.jpg".format(i, f_i))
            cv2.imwrite(fname, 255. * im[:, :, 0])

        #time.sleep(.5)
        #dataset.DisplayData(vdm2)
        for f_i, im in enumerate(vdm2[0]):
            fname = os.path.join(d_name, "forward", "vid_{}_frame_{}.jpg".format(i, f_i))
            cv2.imwrite(fname, 255. * im[:, :, 0])
        print("vid completed")
        #time.sleep(.5)

        for f_i, im in enumerate(vdm3[0]):
            fname = os.path.join(d_name, "backward", "vid_{}_frame_{}.jpg".format(i, f_i))
            cv2.imwrite(fname, 255. * im[:, :, 0])
        print("vid completed")

    s_means = np.array(all_s_means)
    print(s_means.shape)
    if FLAGS.num_features / 2 > 2:
        tsne = TSNE()
        s_feats = tsne.fit_transform(s_feats[:1000])
        tsne = TSNE()
        t_feats = tsne.fit_transform(t_feats[:1000])
        tsne = TSNE()
        s_means = tsne.fit_transform(s_means[:1000])
    print("S means")
    plt.scatter(s_means[:, 0], s_means[:, 1])
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.show()
    #plt.scatter(s_means[:3, 0], s_means[:3, 1], c='r')
    plt.scatter(s_feats[:16, 0], s_feats[:16, 1], c='b')
    plt.scatter(s_feats[16:32, 0], s_feats[16:32, 1], c='b')
    plt.scatter(s_feats[32:48, 0], s_feats[32:48, 1], c='b')
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.show()


    print("S")
    colors = ['ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-']
    for j in range(5):
        st = 16*3*j
        for i in range(6):
            sf = s_feats[st + i*16: st + (i+1)*16]
            color = colors[i]
            x, y = sf[:, 0], sf[:, 1]
            plt.plot(x, y, color)
        # plt.ylim([-3, 3])
        # plt.xlim([-3, 3])
        plt.show()
    print("T")
    colors = ['ro-', 'go-', 'bo-', 'co-', 'mo-', 'yo-']
    for j in range(5):
        st = 16*3*j
        for i in range(6):
            tf = t_feats[st + i*16: st + (i+1)*16]
            color = colors[i]
            x, y = tf[:, 0], tf[:, 1]
            plt.plot(x, y, color)
        # plt.ylim([-3, 3])
        # plt.xlim([-3, 3])
        plt.show()

def generate_visualizations():
    # kinda (totally) hacky
    ls, gs = train(test=True)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    # Run test data through model and collect dataset of S and T features for t-sne
    n_iters = 100
    all_t_feats = []
    all_s_feats = []
    all_s_means = []
    for i in range(n_iters):
        batch = dataset.GET_TEST_BATCH()
        [t_feats, s_feats, vdm] = sess.run(
            [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                videos_placehoder: batch
            }
        )
        #dataset.DisplayData(vdm)

        t_vids = [f for f in t_feats]
        s_vids = [f for f in s_feats]
        s_means = [s_f.mean(axis=0) for s_f in s_vids]

        all_t_feats.extend(t_vids)
        all_s_feats.extend(s_vids)
        all_s_means.extend(s_means)

    t_feats = np.concatenate(all_t_feats)
    s_feats = np.concatenate(all_s_feats)

    # static interpolation
    s_feats_interp = []
    t_feats_interp = []
    for i in range(len(all_s_means) - 1):
        s_cur = all_s_means[i]
        s_next = all_s_means[i+1]
        nf = float(FLAGS.num_frames)
        s_interp = [s_cur * (i / nf) + s_next * (1.0 - (i / nf)) for i in range(int(nf))]
        s_interp = np.array(s_interp)
        print(s_interp.shape, "s_interp")
        rand_tf = random.choice(t_feats)
        t_interp = [rand_tf for i in range(int(nf))]
        t_interp = np.array(t_interp)
        print(t_interp.shape, "t_interp")
        s_feats_interp.append(s_interp)
        t_feats_interp.append(t_interp)
    s_feats_interp = np.concatenate(s_feats_interp)
    t_feats_interp = np.concatenate(t_feats_interp)
    print(s_feats_interp.shape, t_feats_interp.shape)
    f = np.concatenate([s_feats_interp, t_feats_interp], axis=1)

    d_name = os.path.join(FLAGS.train_dir, "static_interp_videos")
    if not os.path.isdir(d_name):
        os.mkdir(d_name)
    for i in range(99):
        [vdm] = sess.run(
            [summary_videos], feed_dict={
                frames_z: f[i*16:(i+1)*16]
            }
        )
        print("s interp vid")
        for f_i, im in enumerate(vdm[0]):
            fname = os.path.join(d_name, "vid_{}_frame_{}.jpg".format(i, f_i))
            cv2.imwrite(fname, 255. * im[:, :, 0])

    # temporal interpolation
    s_feats_interp = []
    t_feats_interp = []
    for i in range(len(all_s_means) - 1):
        s_cur = all_s_means[i]

        nf = float(FLAGS.num_frames)

        s_interp = [s_cur for i in range(int(nf))]
        s_interp = np.array(s_interp)

        t_0, t_end = all_t_feats[i][0], all_t_feats[i][-1]
        t_interp = [t_0 * (i / nf) + t_end * (1.0 - (i / nf)) for i in range(int(nf))]


        s_feats_interp.append(s_interp)
        t_feats_interp.append(t_interp)
    s_feats_interp = np.concatenate(s_feats_interp)
    t_feats_interp = np.concatenate(t_feats_interp)

    f = np.concatenate([s_feats_interp, t_feats_interp], axis=1)

    d_name = os.path.join(FLAGS.train_dir, "temporal_interp_videos")
    if not os.path.isdir(d_name):
        os.mkdir(d_name)
    for i in range(99):
        [vdm] = sess.run(
            [summary_videos], feed_dict={
                frames_z: f[i*16:(i+1)*16]
            }
        )
        print("t interp vid")
        for f_i, im in enumerate(vdm[0]):
            fname = os.path.join(d_name, "vid_{}_frame_{}.jpg".format(i, f_i))
            cv2.imwrite(fname, 255. * im[:, :, 0])



def generate_temporal_test_distributions():
    # kinda (totally) hacky
    ls, gs = train(test=True)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    # Run test data through model and collect dataset of S and T features for t-sne
    n_iters = 3000
    all_t_feats = []
    for i in range(n_iters):
        batch = dataset.GET_TEST_BATCH()
        [t_feats, vdm] = sess.run(
            [videos_mu_t, summary_videos], feed_dict={
                videos_placehoder: batch
            }
        )
        #dataset.DisplayData(vdm)
        #print(t_feats.shape, s_feats.shape)
        t_vids = [f for f in t_feats]

        all_t_feats.extend(t_vids)

    t_feats = np.concatenate(all_t_feats)

    # shifted_t_feats = np.concatenate(all_t_feats[1:] + [all_t_feats[0]])
    # f = np.concatenate([s_feats, t_feats], axis=1)
    # shifted_f = np.concatenate([s_feats, shifted_t_feats], axis=1)
    # print(f.shape)
    # print(shifted_f.shape)
    # for i in range(100):
    #     [vdm] = sess.run(
    #         [summary_videos], feed_dict={
    #             frames_z: f[i*16:(i+1)*16]
    #         }
    #     )
    #     [vdm2] = sess.run(
    #         [summary_videos], feed_dict={
    #             frames_z: shifted_f[i*16:(i+1)*16]
    #         }
    #     )
    #     dataset.DisplayData(vdm)
    #     time.sleep(.5)
    #     dataset.DisplayData(vdm2)
    #     time.sleep(.5)

    if FLAGS.num_features > 2:
        tsne = TSNE()
        t_feats = tsne.fit_transform(t_feats[:1000])

    print("T")
    colors = ['ro-', 'go-', 'bo-']
    for j in range(5):
        st = 16*3*j
        for i in range(3):
            tf = t_feats[st + i*16: st + (i+1)*16]
            color = colors[i]
            x, y = tf[:, 0], tf[:, 1]
            plt.plot(x, y, color)
        # plt.ylim([-3, 3])
        # plt.xlim([-3, 3])
        plt.show()

def test_disentanglement():
    ls, gs = train(test=True)
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']


    dataset = data_handler.BouncingMNISTDataHandler(
        num_frames=1, batch_size=FLAGS.batch_size,
        image_size=FLAGS.im_size, num_digits=1
    )
    def generate_data(train):
        # generate training data
        num_batches = 300
        data = []
        tfeats, sfeats, slabels, tlabels = [], [], [], []
        f = dataset.GetBatch if train else dataset.GetTestBatch
        for i in range(num_batches):
            b, t, s = f(return_classes=True)
            [t_feats, s_feats, vdm] = sess.run(
                [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )

            tfeats.append(t_feats)
            sfeats.append(s_feats)
            tlabels.append(t[0])
            slabels.append(s[0])

        tfeats = np.concatenate(tfeats)[:, 0, :]
        sfeats = np.concatenate(sfeats)[:, 0, :]
        slabels = np.concatenate(slabels, axis=0)
        tlabels = np.concatenate(tlabels, axis=0)
        return tfeats, sfeats, tlabels, slabels
    tfeats, sfeats, tlabels, slabels = generate_data(True)
    tfeats_test, sfeats_test, tlabels_test, slabels_test = generate_data(False)

    # train static -> static
    print(sfeats.shape)
    print(slabels.shape)
    clf = sklearn.svm.SVC()
    print("Fitting static -> static")
    clf.fit(sfeats, slabels)
    ss_score = clf.score(sfeats_test, slabels_test)
    print(ss_score)
    clf = sklearn.svm.SVC()
    print("Fitting temporal -> temporal")
    clf.fit(tfeats, tlabels)
    tt_score = clf.score(tfeats_test, tlabels_test)
    print(tt_score)

    clf = sklearn.svm.SVC()
    print("Fitting static -> temporal")
    clf.fit(sfeats, tlabels)
    st_score = clf.score(sfeats_test, tlabels_test)
    print(st_score)
    clf = sklearn.svm.SVC()
    print("Fitting temporal -> static")
    clf.fit(tfeats, slabels)
    ts_score = clf.score(tfeats_test, slabels_test)
    print(ts_score)

    d_score = d_score = (ss_score * tt_score)**.5 / (st_score * ts_score)**.5
    print("D Score: {}".format(d_score))

    all_feats = np.concatenate([sfeats, tfeats], 1)
    all_feats_test = np.concatenate([sfeats_test, tfeats_test], 1)
    clf = sklearn.svm.SVC()
    print("Fitting all -> static")
    clf.fit(all_feats, slabels)
    score = clf.score(all_feats_test, slabels_test)
    print(score)
    clf = sklearn.svm.SVC()
    print("Fitting all -> temporal")
    clf.fit(all_feats, tlabels)
    score = clf.score(all_feats_test, tlabels_test)
    print(score)

def test_disentanglement_2():
    ls, gs = train(test=True)
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']
    dataset = ls['dataset']


    # id_dataset = data_handler.BouncingMNISTDataHandler(
    #     num_frames=1, batch_size=FLAGS.batch_size,
    #     image_size=FLAGS.im_size, num_digits=1
    # )

    # t_dataset = data_handler.BouncingMNISTDataHandler(
    #     num_frames=FLAGS.num_frames, batch_size=FLAGS.batch_size,
    #     image_size=FLAGS.im_size, num_digits=1
    # )

    # def generate_data(train):
    #     # generate training data
    #     num_batches = 300
    #     data = []
    #     tfeats, sfeats, slabels, tlabels = [], [], [], []
    #     f = id_dataset.GetBatch if train else id_dataset.GetTestBatch
    #     for i in range(num_batches):
    #         b, t, s = f(return_classes=True)
    #         [t_feats, s_feats, vdm] = sess.run(
    #             [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
    #                 videos_placehoder: b
    #             }
    #         )

    #         tfeats.append(t_feats)
    #         sfeats.append(s_feats)
    #         tlabels.append(t[0])
    #         slabels.append(s[0])

    #     tfeats = np.concatenate(tfeats)[:, 0, :]
    #     sfeats = np.concatenate(sfeats)[:, 0, :]
    #     slabels = np.concatenate(slabels, axis=0)
    #     tlabels = np.concatenate(tlabels, axis=0)
    #     return tfeats, sfeats, tlabels, slabels

    def generate_rot_data(train):
        num_batches = 30
        data = []
        tfeats, sfeats= [], []
        f = dataset.GET_BATCH if train else dataset.GET_TEST_BATCH
        for i in range(num_batches):
            b  = f()
            [t_feats, s_feats, vdm] = sess.run(
                [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )

            tfeats.append(t_feats)
            sfeats.append(s_feats)
            #print(tfeats.shape, sfeats.shape)

        tfeats = np.concatenate(tfeats)
        sfeats = np.concatenate(sfeats)

        t_rot_feats, t_rot_labels = generate_chair_rotation_dataset(tfeats)
        s_rot_feats, s_rot_labels = generate_chair_rotation_dataset(sfeats)
        AF = np.concatenate([tfeats, sfeats], axis=2)
        all_rot_feats, all_rot_labels = generate_chair_rotation_dataset(AF)
        return t_rot_feats, t_rot_labels, s_rot_feats, s_rot_labels, all_rot_feats, all_rot_labels

    # tfeats, sfeats, tlabels, slabels = generate_data(True)
    # tfeats_test, sfeats_test, tlabels_test, slabels_test = generate_data(False)


    tfeats_rot, tlabels_rot, sfeats_rot, slabels_rot, all_feats_rot, all_labels_rot = generate_rot_data(True)
    tfeats_test_rot, tlabels_test_rot, sfeats_test_rot, slabels_test_rot, all_feats_test_rot, all_labels_test_rot = generate_rot_data(False)


    # train static -> static
    # print(sfeats.shape)
    # print(slabels.shape)
    # clf = sklearn.svm.SVC()
    # print("Fitting static -> static")
    # clf.fit(sfeats, slabels)
    # ss_score = clf.score(sfeats_test, slabels_test)
    # print(ss_score)
    clf = sklearn.svm.SVC()
    print("Fitting temporal -> temporal")
    clf.fit(tfeats_rot, tlabels_rot)
    tt_score = clf.score(tfeats_test_rot, tlabels_test_rot)
    print(tt_score)

    clf = sklearn.svm.SVC()
    print("Fitting static -> temporal")
    clf.fit(sfeats_rot, slabels_rot)
    st_score = clf.score(sfeats_test_rot, slabels_test_rot)
    print(st_score)
    # clf = sklearn.svm.SVC()
    # print("Fitting temporal -> static")
    # clf.fit(tfeats, slabels)
    # ts_score = clf.score(tfeats_test, slabels_test)
    # print(ts_score)

    # d_score = d_score = (ss_score * tt_score)**.5 / (st_score * ts_score)**.5
    # print("D Score: {}".format(d_score))

    # all_feats = np.concatenate([sfeats, tfeats], 1)
    # all_feats_test = np.concatenate([sfeats_test, tfeats_test], 1)
    # clf = sklearn.svm.SVC()
    # print("Fitting all -> static")
    # clf.fit(all_feats, slabels)
    # score = clf.score(all_feats_test, slabels_test)
    # print(score)
    clf = sklearn.svm.SVC()
    print("Fitting all -> temporal")
    clf.fit(all_feats_rot, all_labels_rot)
    score = clf.score(all_feats_test_rot, all_labels_test_rot)
    print(score)

def test_temporal_disentanglement():
    ls, gs = train(test=True)
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']

    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    dataset = data_handler.BouncingMNISTDataHandler(
        num_frames=1, batch_size=FLAGS.batch_size,
        image_size=FLAGS.im_size, num_digits=1
    )
    def generate_data(train):
        # generate training data
        num_batches = 300
        data = []
        all_feats, slabels, tlabels = [], [], []
        f = dataset.GetBatch if train else dataset.GetTestBatch
        for i in range(num_batches):
            b, t, s = f(return_classes=True)
            [feats, vdm] = sess.run(
                [videos_mu_t, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )

            all_feats.append(feats)
            tlabels.append(t[0])
            slabels.append(s[0])

        all_feats = np.concatenate(all_feats)[:, 0, :]
        slabels = np.concatenate(slabels, axis=0)
        tlabels = np.concatenate(tlabels, axis=0)
        return all_feats, tlabels, slabels
    feats, tlabels, slabels = generate_data(True)
    feats_test, tlabels_test, slabels_test = generate_data(False)

    # train static -> static
    print(feats.shape)
    print(slabels.shape)
    clf = sklearn.svm.SVC()
    print("Fitting all -> static")
    clf.fit(feats, slabels)
    score = clf.score(feats_test, slabels_test)
    print(score)
    clf = sklearn.svm.SVC()
    print("Fitting all -> temporal")
    clf.fit(feats, tlabels)
    score = clf.score(feats_test, tlabels_test)
    print(score)

    # get combinations
    l = set(range(feats.shape[1]))
    d_scores = []
    for s_inds in itertools.chain(itertools.combinations(l, 2), itertools.combinations(l, 1), itertools.combinations(l, 3)):
        t_inds = list(l - set(s_inds))

        sfeats = feats[:, s_inds]
        tfeats = feats[:, t_inds]

        sfeats_test = feats_test[:, s_inds]
        tfeats_test = feats_test[:, t_inds]
        print("S feats: {} | T feats: {}".format(s_inds, t_inds))

        clf = sklearn.svm.SVC()
        print("Fitting static -> static")
        clf.fit(sfeats, slabels)
        ss_score = clf.score(sfeats_test, slabels_test)
        print(ss_score)
        clf = sklearn.svm.SVC()
        print("Fitting temporal -> temporal")
        clf.fit(tfeats, tlabels)
        tt_score = clf.score(tfeats_test, tlabels_test)
        print(tt_score)

        clf = sklearn.svm.SVC()
        print("Fitting static -> temporal")
        clf.fit(sfeats, tlabels)
        st_score = clf.score(sfeats_test, tlabels_test)
        print(st_score)
        clf = sklearn.svm.SVC()
        print("Fitting temporal -> static")
        clf.fit(tfeats, slabels)
        ts_score = clf.score(tfeats_test, slabels_test)
        print(ts_score)

        d_score = (ss_score * tt_score)**.5 / (st_score * ts_score)**.5
        print("Disentanglement Score: {}".format(d_score))
        d_scores.append(d_score)

    print("Max D Score: {}".format(max(d_scores)))

def test_temporal_chair_disentanglement():
    ls, gs = train(test=True)
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']

    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_mu = ls['frames_mu']
    dataset = ls['dataset']

    id_dataset = data_handler.ChairTestDataset(
        FLAGS.batch_size, FLAGS.num_frames,
        os.path.join(FLAGS.data_folder, "test_id"),
        FLAGS.im_size, greyscale=True
    )

    def generate_id_data(train):
        # generate training data
        num_batches = 10#300
        data = []
        all_feats, id_labels = [], []
        for i in range(num_batches):
            b, ls = id_dataset.get_id_batch(train)
            [feats, vdm] = sess.run(
                [frames_mu, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )
            all_feats.append(feats)

            id_labels.extend(ls)

            print(i)
        all_feats = np.concatenate(all_feats)#[:, 0, :]
        slabels = np.concatenate(id_labels)


        return all_feats, slabels
    feats, slabels = generate_id_data(True)
    feats_test, slabels_test = generate_id_data(False)

    def generate_rot_data(train, inds):
        num_batches = 30
        data = []
        all_feats = []
        f = dataset.GET_BATCH if train else dataset.GET_TEST_BATCH
        for i in range(num_batches):
            b  = f()
            [feats, vdm] = sess.run(
                [frames_mu, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )
            feats = feats.reshape([FLAGS.batch_size, FLAGS.num_frames, FLAGS.num_features])

            all_feats.append(feats[:, :, inds])


        all_feats = np.concatenate(all_feats)
        print(all_feats.shape)
        rot_feats, rot_labels = generate_chair_rotation_dataset(all_feats)

        return rot_feats, rot_labels

    # train static -> static
    print(feats.shape)
    print(slabels.shape)
    clf = sklearn.svm.SVC()
    print("Fitting all -> static")
    clf.fit(feats, slabels)
    score = clf.score(feats_test, slabels_test)
    print(score)

    all_t_feats, tlabels = generate_rot_data(True, range(FLAGS.num_features))
    all_t_feats_test, tlabels_test = generate_rot_data(False, range(FLAGS.num_features))
    print(all_t_feats.shape)
    clf = sklearn.svm.SVC()
    print("Fitting all -> temporal")
    clf.fit(all_t_feats, tlabels)
    score = clf.score(all_t_feats_test, tlabels_test)
    print(score)


    # get combinations
    l = set(range(feats.shape[1]))
    d_scores = []
    for s_inds in itertools.chain(itertools.combinations(l, 2), itertools.combinations(l, 1), itertools.combinations(l, 3), itertools.combinations(l, 4)):
        t_inds = list(l - set(s_inds))

        sfeats = feats[:, s_inds]
        tfeats, tlabels = generate_rot_data(True, t_inds)
        t0 = [t for t in t_inds]
        t1 = [t + FLAGS.num_features for t in t_inds]
        t2 = [t + 2*FLAGS.num_features for t in t_inds]
        all_tinds = t0 + t1 + t2

        t0 = [t for t in s_inds]
        t1 = [t + FLAGS.num_features for t in s_inds]
        t2 = [t + 2*FLAGS.num_features for t in s_inds]
        all_sinds = t0 + t1 + t2

        tfeats = all_t_feats[:, all_tinds]

        sfeats_test = feats_test[:, s_inds]
        tfeats_test = all_t_feats_test[:, all_tinds]

        tfeats_swap = feats[:, t_inds]
        tfeats_swap_test = feats_test[:, t_inds]

        sfeats_swap = all_t_feats[:, all_sinds]
        sfeats_swap_test = all_t_feats_test[:, all_sinds]

        print("S feats: {} | T feats: {}".format(s_inds, t_inds))

        clf = sklearn.svm.SVC()
        print("Fitting static -> static")
        clf.fit(sfeats, slabels)
        ss_score = clf.score(sfeats_test, slabels_test)
        print(ss_score)
        clf = sklearn.svm.SVC()
        print("Fitting temporal -> temporal")
        clf.fit(tfeats, tlabels)
        tt_score = clf.score(tfeats_test, tlabels_test)
        print(tt_score)

        clf = sklearn.svm.SVC()
        print("Fitting static -> temporal")
        clf.fit(sfeats_swap, tlabels)
        st_score = clf.score(sfeats_swap_test, tlabels_test)
        print(st_score)
        clf = sklearn.svm.SVC()
        print("Fitting temporal -> static")
        clf.fit(tfeats_swap, slabels)
        ts_score = clf.score(tfeats_swap_test, slabels_test)
        print(ts_score)

        d_score = (ss_score * tt_score)**.5 / (st_score * ts_score)**.5
        print("Disentanglement Score: {}".format(d_score))
        d_scores.append(d_score)
        d_scores.append(1.0 / d_score)

    print("Max D Score: {}".format(max(d_scores)))

def test_chair_disentanglement():
    ls, gs = train(test=True)
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    dataset = ls['dataset']
    videos_placehoder = ls['videos_placeholder']
    rot_dataset = ls["dataset"]
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']


    id_dataset = data_handler.ChairTestDataset(
        FLAGS.batch_size, FLAGS.num_frames,
        os.path.join(FLAGS.data_folder, "test_id"),
        FLAGS.im_size, greyscale=True
    )

    def generate_id_data(train):
        # generate training data
        num_batches = 10
        data = []
        tfeats, sfeats, id_labels = [], [], []
        for i in range(num_batches):
            b, ls = id_dataset.get_id_batch(train)
            [t_feats, s_feats, vdm] = sess.run(
                [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )
            t_feats = t_feats.reshape([FLAGS.batch_size * FLAGS.num_frames, -1])
            s_feats = s_feats.reshape([FLAGS.batch_size * FLAGS.num_frames, -1])
            ls = ls.reshape([FLAGS.batch_size * FLAGS.num_frames, -1])

            tfeats.append(t_feats)
            sfeats.append(s_feats)
            id_labels.append(ls)

        tfeats = np.concatenate(tfeats)
        sfeats = np.concatenate(sfeats)
        id_labels = np.concatenate(id_labels, axis=0)[:, 0]
        return tfeats, sfeats, id_labels

    def generate_rot_data(train):
        num_batches = 30
        data = []
        tfeats, sfeats= [], []
        f = dataset.GET_BATCH if train else dataset.GET_TEST_BATCH
        for i in range(num_batches):
            b  = f()
            [t_feats, s_feats, vdm] = sess.run(
                [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                    videos_placehoder: b
                }
            )

            tfeats.append(t_feats)
            sfeats.append(s_feats)
            #print(tfeats.shape, sfeats.shape)

        tfeats = np.concatenate(tfeats)
        sfeats = np.concatenate(sfeats)

        t_rot_feats, t_rot_labels = generate_chair_rotation_dataset(tfeats)
        s_rot_feats, s_rot_labels = generate_chair_rotation_dataset(sfeats)
        AF = np.concatenate([tfeats, sfeats], axis=2)
        all_rot_feats, all_rot_labels = generate_chair_rotation_dataset(AF)
        return t_rot_feats, t_rot_labels, s_rot_feats, s_rot_labels, all_rot_feats, all_rot_labels

    tfeats_id, sfeats_id, labels_id = generate_id_data(True)
    all_feats_id = np.concatenate([tfeats_id, sfeats_id], axis=1)
    tfeats_test_id, sfeats_test_id, labels_test_id = generate_id_data(False)
    all_feats_test_id = np.concatenate([tfeats_test_id, sfeats_test_id], axis=1)

    tfeats_rot, tlabels_rot, sfeats_rot, slabels_rot, all_feats_rot, all_labels_rot = generate_rot_data(True)
    tfeats_test_rot, tlabels_test_rot, sfeats_test_rot, slabels_test_rot, all_feats_test_rot, all_labels_test_rot = generate_rot_data(True)

    # train static -> static
    clf = sklearn.svm.SVC()
    print("Fitting static -> static")
    clf.fit(sfeats_id, labels_id)
    ss_score = clf.score(sfeats_test_id, labels_test_id)
    print(ss_score)

    clf = sklearn.svm.SVC()
    print("Fitting temporal -> temporal")
    clf.fit(tfeats_rot, tlabels_rot)
    tt_score = clf.score(tfeats_test_rot, tlabels_test_rot)
    print(tt_score)

    clf = sklearn.svm.SVC()
    print("Fitting static -> temporal")
    clf.fit(sfeats_rot, slabels_rot)
    st_score = clf.score(sfeats_test_rot, slabels_test_rot)
    print(st_score)

    clf = sklearn.svm.SVC()
    print("Fitting temporal -> static")
    clf.fit(tfeats_id, labels_id)
    ts_score = clf.score(tfeats_test_id, labels_test_id)
    print(ts_score)

    d_score = d_score = (ss_score * tt_score)**.5 / (st_score * ts_score)**.5
    print("D Score: {}".format(d_score))

    clf = sklearn.svm.SVC()
    print("Fitting all -> static")
    clf.fit(all_feats_id, labels_id)
    score = clf.score(all_feats_test_id, labels_test_id)
    print(score)
    clf = sklearn.svm.SVC()
    print("Fitting all -> temporal")
    clf.fit(all_feats_rot, all_labels_rot)
    score = clf.score(all_feats_test_rot, all_labels_test_rot)
    print(score)

def generate_chair_test_distributions():
    # kinda (totally) hacky
    ls, gs = train(test=True)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    # Run test data through model and collect dataset of S and T features for t-sne
    n_iters = 100
    all_t_feats = []
    all_s_feats = []
    all_s_means = []
    for i in range(n_iters):
        batch = dataset.GET_TEST_BATCH()
        [t_feats, s_feats, vdm] = sess.run(
            [videos_mu_t, videos_mu_s, summary_videos], feed_dict={
                videos_placehoder: batch
            }
        )
        #dataset.DisplayData(vdm)

        t_vids = [f for f in t_feats]
        s_vids = [f for f in s_feats]
        s_means = [s_f.mean(axis=0) for s_f in s_vids]

        all_t_feats.extend(t_vids)
        all_s_feats.extend(s_vids)
        all_s_means.extend(s_means)

    t_feats = np.concatenate(all_t_feats)
    s_feats = np.concatenate(all_s_feats)
    shifted_t_feats = np.concatenate(all_t_feats[1:] + [all_t_feats[0]])
    f = np.concatenate([s_feats, t_feats], axis=1)
    shifted_f = np.concatenate([s_feats, shifted_t_feats], axis=1)
    print(f.shape)
    print(shifted_f.shape)

    # for i in range(10):
    #     [vdm] = sess.run(
    #         [summary_videos], feed_dict={
    #             frames_z: f[i*16:(i+1)*16]
    #         }
    #     )
    #     [vdm2] = sess.run(
    #         [summary_videos], feed_dict={
    #             frames_z: shifted_f[i*16:(i+1)*16]
    #         }
    #     )
    #     dataset.DisplayData(vdm)
    #     time.sleep(.5)
    #     dataset.DisplayData(vdm2)
    #     time.sleep(.5)


    s_means = np.array(all_s_means)
    print(s_means.shape)
    if FLAGS.num_features / 2 > 2:
        tsne = TSNE(early_exaggeration=1.)
        s_feats = tsne.fit_transform(s_feats[:1000])
        tsne = TSNE(early_exaggeration=1.)
        t_feats = tsne.fit_transform(t_feats[:1000])
        tsne = TSNE(early_exaggeration=1.)
        s_means = tsne.fit_transform(s_means[:1000])
        # s_feats = s_feats[:, [0, 1]]
        # t_feats = t_feats[:, [0, 1]]
        # s_means = s_means[:, [0, 1]]
    print("S means")
    plt.scatter(s_means[:, 0], s_means[:, 1])
    # plt.ylim([-3, 3])
    # plt.xlim([-3, 3])
    plt.show()
    #plt.scatter(s_means[:3, 0], s_means[:3, 1], c='r')
    plt.scatter(s_feats[:16, 0], s_feats[:16, 1], c='b')
    plt.scatter(s_feats[16:32, 0], s_feats[16:32, 1], c='b')
    plt.scatter(s_feats[32:48, 0], s_feats[32:48, 1], c='b')
    # plt.ylim([-3, 3])
    # plt.xlim([-3, 3])
    plt.show()


    print("S")
    colors = ['ro-', 'go-', 'bo-']
    for j in range(5):
        st = 16*3*j
        for i in range(3):
            sf = s_feats[st + i*16: st + (i+1)*16]
            color = colors[i]
            x, y = sf[:, 0], sf[:, 1]
            plt.plot(x, y, color)
        # plt.ylim([-3, 3])
        # plt.xlim([-3, 3])
        plt.show()
    print("T")
    colors = ['ro-', 'go-', 'bo-']
    for j in range(5):
        st = 16*3*j
        for i in range(3):
            tf = t_feats[st + i*16: st + (i+1)*16]
            color = colors[i]
            x, y = tf[:, 0], tf[:, 1]
            plt.plot(x, y, color)
        # plt.ylim([-3, 3])
        # plt.xlim([-3, 3])
        plt.show()


def run_test(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    if FLAGS.dataset == "chairs":
        #if FLAGS.num_frames == 1:

        #generate_visualizations()
        if FLAGS.temporal_only:
            test_temporal_chair_disentanglement()
        else:
            test_chair_disentanglement()
        #else:
        #generate_chair_test_distributions()
    else:
        if FLAGS.temporal_only:
            if FLAGS.num_frames == 1:
                test_temporal_disentanglement()
            else:
                generate_temporal_test_distributions()
        else:
            if FLAGS.num_frames == 1:
                test_disentanglement()
            elif FLAGS.batch_size == 1:
                generate_visualizations()
                generate_test_distributions()
            else:
                # test with different temporal task
                test_disentanglement_2()



if __name__ == '__main__':
    tf.app.run(main=run_test)




