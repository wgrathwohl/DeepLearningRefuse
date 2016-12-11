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

from video_variational_autoencoder import *
import cv2
import time
from sklearn.manifold import TSNE
import sklearn.svm
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import cv2

def generate_temporal_test_distributions():
    # kinda (totally) hacky
    ls, gs = train(False)
    dataset = ls['dataset']
    sess = ls['sess']
    encoding_mu = ls['encoding_mu']
    images_placehoder = ls['images_placeholder']
    dec_mu = ls['decoded_images']


    # Run test data through model and collect dataset of S and T features for t-sne
    n_iters = 300
    all_t_feats = []
    for i in range(n_iters):
        batch = np.concatenate(dataset.GetTestBatch()[0], 0)
        [feats] = sess.run(
            [encoding_mu], feed_dict={
                images_placehoder: batch
            }
        )
        all_t_feats.extend(feats)

    t_feats = np.concatenate(all_t_feats)

    print(t_feats)
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
    ls, gs = train(False)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']
    videos_mu_s = ls['videos_mu_s']
    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    dataset = data_handler.BouncingMNISTDataHandler(
        num_frames=1, batch_size=64,
        image_size=FLAGS.im_size, num_digits=1
    )
    def generate_data(train):
        # generate training data
        num_batches = 100
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

def test_temporal_disentanglement():
    ls, gs = train(False)
    dataset = ls['dataset']
    sess = ls['sess']
    videos_mu_t = ls['videos_mu_t']

    videos_placehoder = ls['videos_placeholder']
    videos_dec_mu = ls['videos_dec_mu']
    summary_videos = ls['summary_videos']
    frames_z = ls['frames_z']

    dataset = data_handler.BouncingMNISTDataHandler(
        num_frames=1, batch_size=64,
        image_size=FLAGS.im_size, num_digits=1
    )
    def generate_data(train):
        # generate training data
        num_batches = 100
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
    for s_inds in itertools.combinations(l, 2):
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


def run_test(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    #test_temporal_disentanglement()
    generate_temporal_test_distributions()



if __name__ == '__main__':
    tf.app.run(main=run_test)
