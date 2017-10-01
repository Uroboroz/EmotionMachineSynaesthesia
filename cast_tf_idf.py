import psycopg2
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import imageio
import glob
from sys import argv
from pickle import load, dump
from os import _exit


# python cast_tf_idf.py --path_file test.txt --save_tfidf 1
# --host_db localhost --user_db postgres --pass_db uouwrecr13
# --db_name postgres --feedback feedback_2.csv


db = psycopg2.connect()
cursor = db.cursor()


class SOM:
    N = 0
    n_teacher = 0

    def create_picture_NMF(self, node, name):
        nodes = []
        for i in range(self.N):
            model = NMF(n_components=3)
            model.fit_transform(node[i])
            H = model.components_
            nodes.append([[H[0][j], H[1][j], H[2][j]] for j in range(self.N)])
        plt.imshow(nodes, interpolation='none')
        plt.savefig("./image/" + name + ".png")

    def __init__(self, teachers_, N):
        self.n_teacher = len(teachers_)
        self.N = N
        teachers = np.array(teachers_)
        nodes = np.random.rand(self.N, self.N, self.N)  # node array. each node has 3-dim weight vector
        self.create_picture_NMF(nodes, "start")

        for i in range(len(teachers)):
            self.train(nodes, teachers, i)
            self.create_picture_NMF(nodes, str(i))

        # output
        self.create_picture_NMF(nodes, "finish")
        with imageio.get_writer('./image/movie.gif', mode='I') as writer:
            for filename in glob.glob('./image/*.png'):
                image = imageio.imread(filename)
                writer.append_data(image)

    def train(self, nodes, teachers, i):
        bmu = self.best_matching_unit(nodes, teachers[i])
        # print bmu
        for x in range(self.N):
            for y in range(self.N):
                c = np.array([x, y])  # coordinate of unit
                d = np.linalg.norm(c - bmu)
                L = self.learning_ratio(i)
                S = self.learning_radius(i, d)
                for z in range(3):  # TODO clear up using numpy function
                    nodes[x, y, z] += L * S * (teachers[i, z] - nodes[x, y, z])

    def best_matching_unit(self, nodes, teacher):
        # TODO simplify using numpy function
        norms = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                for k in range(3):
                    norms[i, j] += (nodes[i, j, k] - teacher[k]) ** 2
        bmu = np.argmin(norms)
        return np.unravel_index(bmu, (self.N, self.N))

    def neighbourhood(self, t):
        halflife = float(self.n_teacher / 4)
        initial = float(self.N / 2)
        return initial * np.exp(-t / halflife)

    def learning_ratio(self, t):
        halflife = float(self.n_teacher / 4)
        initial = 0.1
        return initial * np.exp(-t / halflife)

    def learning_radius(self, t, d):
        s = self.neighbourhood(t)
        return np.exp(-d ** 2 / (2 * s ** 2))


class EmotionTFIDF:
    corpus = []
    size = 0

    def __init__(self, path_data_learning, flag_save, flag_load=False):
        print("Start load corpus...")
        if flag_load:
            with open(path_data_learning) as tfidf_dump:
                self.corpus = load(tfidf_dump)
                tfidf_dump.close()
        else:
            with open(path_data_learning, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    self.corpus.append(
                        [remove_p(str(download_comment(row[0], row[1])[0]).lower()).split(), row[6:18:1]])
                    for word in self.corpus[-1][0]:
                        load_word(word, self.corpus[-1][1])
            self.size = len(self.corpus)

        if flag_save:
            with open("./tfidf.dump", "bw+") as file:
                dump(self.corpus, file)

        print("Finish load corpus.")

    def tf_idf(self, text):
        mtr = []
        for word in remove_p(text).split():
            tf_ = self.tf(word, text)
            idf_ = self.idf(word)
            mtr.append(np.array([tf_[i] * idf_[i] for i in range(11)]))
        return mtr

    @staticmethod
    def emtoion_vector_word(word):
        cursor.execute("SELECT surprise, sadness, anger, disgust, " +
                       "contempt, grief_suffering, shame, interest_excitement, guilt, " +
                       "confusion, GLADNESS FROM mass_media_newsfeed.words_emotions WHERE word = '" + word + "';")
        temp_vector_emotion = cursor.fetchone()
        if temp_vector_emotion is None:
            return [0] * 11
        else:
            return temp_vector_emotion

    def tf(self, word, text):
        return [text.count(word) * int(i) / (len(text) * 10) for i in self.emtoion_vector_word(word)]

    def idf(self, word):
        try:
            return [math.log10((10 * self.size) / int(i)) for i in self.emtoion_vector_word(word)]
        except ZeroDivisionError:
            return [0] * 11


def download_comment(id_comment, id_post):
    cursor.execute("SELECT comments.text " + \
                   "FROM mass_media_newsfeed.comments " + \
                   "WHERE post_id = " + id_post + \
                   "  AND comment_id = " + id_comment + ";")
    return cursor.fetchone()


def load_word(word, em):
    cursor.execute("SELECT mass_media_newsfeed.add_word('" +
                   word + "'::text, " + ", ".join(em) + ");")
    db.commit()
    return


def remove_p(text):
    t = text
    for i in "1234567890-=_+!@#$%^&*()~`,./;\'[]{}:\"<>?|\\":
        t = t.replace(i, " ")
    return t


def __main__(tfidf):
    print("Start tf-idf..")
    try:
        teacher_list = tfidf.tf_idf(open(argv[argv.index('--path_file') + 1], "r").read().lower())
    except FileNotFoundError:
        print("File not exist!")
        _exit(1)
    
    print("Start Self-Organaizing Map")
    SOM(teacher_list, 100)


def print_help():
    print('python3.* cast_tf_idf.py [--path_file _path_file_]\n'
          '                      [--host_db _host_db_  --user_db _user_db_  --pass_db _pass_db_ --db_name _db_name_]\n'
          '                      [--feedback _feedback_ | --load_tfidf _load_tfidf_ ]\n'
          '                      {--save_tfidf 1|0}\n\n'
          '--help:       вызов помощи;\n'
          '--path_file:  путь к файлу, содержайщий данные для анализа;\n'
          '--load_tfidf: путь к файлу для загрузки уже созданной матрицы;\n'
          '--feedback:   путь к файлу, содержащему корпус материалов для обработки данных;\n'
          '--save_tfidf: флаг для сохранения матрица TF-IDF в файл tfidf.dump;\n'
          '--host_db:    адрес БД;\n'
          '--user_db:    имя пользователя БД;\n'
          '--pass_db:    пароль пользователя;\n'
          '--db_name:    имя БД;\n'
          )

try:
    if '--help' in argv:
        print_help()
    elif '--path_file' in argv:
        if '--load_tfidf' in argv:
            __main__(EmotionTFIDF(argv[argv.index('--load_tfidf') + 1], False, flag_load=True))
        if '--host_db' in argv and '--user_db' in argv and '--pass_db' in argv and '--db_name' in argv:
            db = psycopg2.connect(host=argv[argv.index('--host_db') + 1],
                                  user=argv[argv.index('--user_db') + 1],
                                  password=argv[argv.index('--pass_db') + 1],
                                  dbname=argv[argv.index('--db_name') + 1])
            if '--feedback' in argv:
                if '--save_tfidf' in argv and argv[argv.index('--save_tfidf') + 1] == '1':
                    __main__(EmotionTFIDF(argv[argv.index('--feedback') + 1], True))
                else:
                    __main__(EmotionTFIDF(argv[argv.index('--feedback') + 1], False))
            else:
                print_help()
        else:
            print_help()
    else:
        print_help()
except():
    pass
