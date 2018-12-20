import os
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Trainer:
    """Trainer object.

    Attributes
    ----------
    learning_rate : float
        Initial learning rate value

    learning_rate_decay : float
        Decay value of learning rate

    reinit_adam_after_n_batches : int
        Value for reinitialization of Adam optimizer

    batch_size : int

    epochs : int

    best_ckpt_dir : str
        Folder to best checkpoint

    save_dir : str
        Folder for tmp ckpt

    start_from_best : bool
        Warm starting from previous best ckpt

    thresh_plot_cnt: int
        How much thresholds to plot

    save_each : int
        How often to save checkpoints

    input_placeholder : tf.placeholder
        tf.placeholder for data feeding

    quantized_output : tf.Tensor

    """
    def __init__(self,
                 train_graph,
                 data_generator_train,
                 data_generator_valid,
                 input_placeholder,
                 original_output,
                 quantized_output,
                 thresh_plot_cnt=10,
                 save_each=7,
                 start_from_best=False,
                 **kwargs):
        """
        Parameters
        ------
        train_graph : tf.Graph
            Graph with previously created float and quantized model

        data_generator_train : DataGenerator
            Class with trainable data generator creation

        data_generator_valid : DataGenerator
            Class with validation data generator creation

        input_placeholder : tf.placeholder

        original_output : tf.Tensor
            Output of float network

        quantized_output : tf.Tensor
            Output of quantized network

        thresh_plot_cnt : int

        save_each : int

        start_from_best : bool

        kwargs : dict
        """
        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.reinit_adam_after_n_batches = kwargs["reinit_adam_after_n_batches"]
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.best_ckpt_dir = kwargs["best_ckpt_dir"]
        self.save_dir = kwargs["save_dir"]
        self.start_from_best = start_from_best
        self.thresh_plot_cnt = thresh_plot_cnt
        self.save_each = save_each

        # self.logger = logger
        self.__data_generator_train = data_generator_train
        self.__data_generator_valid = data_generator_valid
        self.input_placeholder = input_placeholder
        self.quantized_output = quantized_output

        self._quant_argmax = tf.argmax(self.quantized_output[:, 1:], axis=-1)
        self._original_argmax = tf.argmax(original_output[:, 1:], axis=-1)

        with train_graph.as_default():
            self._build_train_procedure(original_output, quantized_output, self.learning_rate)
            self._build_summaries()

    def _build_train_procedure(self, original_output: tf.Tensor, quantized_output: tf.Tensor, initial_lr):
        loss = (quantized_output - original_output) ** 2
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(loss + 1e-5), axis=-1))
        self.lr = tf.placeholder_with_default(initial_lr, shape=[], name="learning_rate")
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)

    def _build_summaries(self):
        tf.summary.scalar('a_train_loss', self.loss)
        tf.summary.scalar('a_learning_rate', self.lr)

        for t in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.scalar(t.name.replace(":", "_"), t)

        threshold_summary_op = tf.summary.merge_all()

        self.summary_node = threshold_summary_op

    def train(self, sess):
        """Train the model linked to the trainer.

        Parameters
        ----------
        sess : tf.Session
            A session associated with the graph containing the model.
            We delegate responsibility to create a session to user, in order to perform all
            initialization before the training process.
        """

        thresholds = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        optimizer_vars = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - set(thresholds))

        saver = tf.train.Saver(max_to_keep=7)
        loss_min_train = np.Infinity
        loss_min_val = np.Infinity
        filewriter = tf.summary.FileWriter(self.save_dir, sess.graph)
        best_ckpt_path = os.path.join(self.best_ckpt_dir, "best_ckpt.meta")

        if self.start_from_best and os.path.exists(best_ckpt_path):
            saver.restore(sess, best_ckpt_path)

        elif not os.path.exists(self.best_ckpt_dir):
            os.mkdir(self.best_ckpt_dir)

        sess.run([var.initializer for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        global_step_train = 0
        decay_step = 0

        for cur_epoch in range(self.epochs):
            print('')
            print('=' * 40)
            print("Epoch: {}".format(cur_epoch + 1))
            print('=' * 40)

            global_step_train, decay_step, epoch_mean_loss = self._train_epoch(sess,
                                                                               global_step_train, decay_step,
                                                                               thresholds, saver, filewriter,
                                                                               optimizer_vars)

            if epoch_mean_loss < loss_min_train:
                loss_min_train = epoch_mean_loss

            print("    Train loss value: {}, min is {}".format(epoch_mean_loss, loss_min_train))

            # ======================================
            print("Check current accuracy ...")
            top_1_acc, valid_mean_loss = self.validate(sess)

            if self.best_ckpt_dir is not None:
                if valid_mean_loss <= loss_min_val:
                    saver.save(sess, os.path.join(self.best_ckpt_dir, "ckpt_loss_{}".format(int(valid_mean_loss))))
                    loss_min_val = valid_mean_loss

            print("    Valid loss value: {}, min is {}".format(valid_mean_loss, loss_min_val))
            print("    Top 1 acc: {}".format(top_1_acc))
            # ======================================

    def _train_epoch(
            self,
            sess,
            global_step_train,
            decay_step,
            thresholds,
            saver,
            filewriter: tf.summary.FileWriter,
            optimizer_vars):

        loss_arr_train = []

        def _any_is_nan(ths_to_check):
            return any(np.isnan(th) for th in ths_to_check)

        time.sleep(1)  # prevent overlapping between the output of tqdm and the standard stream output

        for batch, _ in tqdm(self.__data_generator_train.generate_batches(self.batch_size)):

            learning_rate_val = self.learning_rate * \
                          np.exp(-global_step_train * self.learning_rate_decay) * \
                          np.abs(np.cos(np.pi * decay_step / 4 / self.reinit_adam_after_n_batches)) + 10.0 ** -7

            loss_value, summary_node_val, ths_vals = self._train_step(sess, batch, learning_rate_val, thresholds)
            loss_arr_train.append(loss_value)

            filewriter.add_summary(summary_node_val, global_step_train)

            global_step_train += 1
            decay_step += 1

            if _any_is_nan(ths_vals):
                print("Some thresholds is None, restore previous trainable value")
                saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
                init_opt_vars_op = tf.variables_initializer(optimizer_vars)
                sess.run(init_opt_vars_op)
            else:
                if global_step_train % self.save_each == 0:
                    saver.save(sess, os.path.join(self.save_dir, "ckpt{}".format(global_step_train)))

            if global_step_train % self.reinit_adam_after_n_batches == 0:
                init_opt_vars_op = tf.variables_initializer(optimizer_vars)
                sess.run(init_opt_vars_op)
                decay_step = 0

        time.sleep(1)  # prevent overlapping between the output of tqdm and the standard stream output

        return global_step_train, decay_step, float(np.mean(loss_arr_train))

    def _train_step(self, sess, im_batch, learning_rate_val, ths):

        feed_dict = {self.lr: learning_rate_val,
                     self.input_placeholder: im_batch}

        _, loss_value, summary_node_val = sess.run([self.train_op, self.loss, self.summary_node], feed_dict)
        ths_vals = sess.run(ths)

        return loss_value, summary_node_val, ths_vals

    def validate(self, sess, use_quantized=True):
        """Check the accuracy of the quantized model linked to the trainer.

        Parameters
        ----------
        sess : tf.Session
            A session associated with the graph containing the model.
            We delegate responsibility to create a session to user, in order to perform all
            initialization before the training process.
        use_quantized: bool
            Indicates whether to use the output of the quantized model or the original one.
        Returns
        -------
        mean_top_1acc: float
            Top 1 accuracy
        mean_L2_valid_loss: float
            The validation loss calculated using the output of the original and quantize models.
        """
        loss_arr_validation = []
        top_1_acc_arr = []

        time.sleep(1)  # prevent overlapping between the output of tqdm and the standard stream output

        for image_batch, label in tqdm(self.__data_generator_valid.generate_batches(self.batch_size)):
            loss_value, pred = self._valid_step(sess, image_batch, use_quantized)
            loss_arr_validation.append(loss_value)
            top_1_acc_arr.append(np.mean(np.equal(pred, label)))

        time.sleep(1)  # prevent overlapping between the output of tqdm and the standard stream output

        return float(np.mean(top_1_acc_arr)), float(np.mean(loss_arr_validation))

    def _valid_step(self, sess, im_batch, use_quantized: bool):
        if use_quantized:
            res = sess.run([self.loss, self._quant_argmax], {self.input_placeholder: im_batch})
        else:
            res = sess.run([self.loss, self._original_argmax], {self.input_placeholder: im_batch})

        return res
