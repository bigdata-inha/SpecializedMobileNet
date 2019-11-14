import pathlib
import numpy as np
import tensorflow as tf


def load_imagenet_tf_dataset_part(directory, batch_size, part_labels, train_cache=True, test_cache=True, data_format="channels_last"):
    """
    imagenet 데이터의 특정 카테고리만을 tf.data.Dataset 형태로 불러옵니다
    """

    data_path = pathlib.Path(directory)

    for path in data_path.glob('*'):
        if "train" in str(path):
            train_path = path
        elif "val" or "test" in str(path):
            test_path = path

    all_train_list = np.array([item for item in train_path.glob('*')])
    all_test_list = np.array([item for item in test_path.glob('*')])

    part_train_path = all_train_list[part_labels]
    part_test_path = all_test_list[part_labels]
    others_labels = np.arange(len(all_test_list)) != part_labels

    others_test_data_path = []
    for category_path in all_test_list[others_labels]:   # 특정 레이블을 제외한 나머지
        others_test_data_path.append(list(category_path.glob('*')))

    others_test_data_path = np.array(others_test_data_path).reshape(-1)
    others_test_data_path = np.random.choice(others_test_data_path, 50, replace=False)

    part_train_path_dataset = [bytes(byte_path.as_posix().encode()) for byte_path in part_train_path.glob('*')]
    part_train_path_dataset = tf.data.Dataset.from_tensor_slices(part_train_path_dataset)

    part_test_data_path_list = list(part_test_path.glob('*'))
    for path in others_test_data_path:
        part_test_data_path_list.append(path)

    part_test_path_dataset = [bytes(byte_path.as_posix().encode()) for byte_path in part_test_data_path_list]
    part_test_path_dataset = tf.data.Dataset.from_tensor_slices(part_test_path_dataset)

    CLASS_NAMES = np.array([item.name for item in train_path.glob('*')])
    class_names = {}
    for i, name in enumerate(CLASS_NAMES):
        class_names[name] = i

    def get_label(file_path):
        parts = tf.strings.split(file_path, '/')
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    @tf.function
    def process_train_dataset(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        # transform the image
        img = tf.image.random_crop(img, (224, 224, 3))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if data_format == "channels_first":
            img = tf.transpose(img, perm=[0, 3, 1, 2])
        return img, label

    @tf.function
    def process_test_dataset(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        # transform the image
        img = tf.image.central_crop(img, 0.875)  # 256 x 256 -> 224 x 224
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if data_format == "channels_first":
            img = tf.transpose(img, perm=[0, 3, 1, 2])
        return img, label

    def prepare_for_training(dataset, batch_size, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                dataset = dataset.cache(cache)
            else:
                dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=-1)
        return dataset

    part_train_dataset = part_train_path_dataset.map(process_train_dataset)
    part_test_dataset = part_test_path_dataset.map(process_test_dataset)

    part_train_dataset = prepare_for_training(part_train_dataset, batch_size, train_cache, shuffle_buffer_size=1300)
    part_test_dataset = prepare_for_training(part_test_dataset, batch_size, test_cache, shuffle_buffer_size=100)

    return part_train_dataset, part_test_dataset