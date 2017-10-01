# EmotionMachineSynaesthesia
EmotionMachineSynaesthesia

python3.* cast_tf_idf.py [--path_file _path_file_]
                      [--host_db _host_db_  --user_db _user_db_  --pass_db _pass_db_ --db_name _db_name_]                    
                      [--feedback _feedback_ | --load_tfidf _load_tfidf_ ][--save_tfidf 1|0]

--help:       вызов помощи;

--path_file:  путь к файлу, содержайщий данные для анализа;

--load_tfidf: путь к файлу для загрузки уже созданной матрицы;

--feedback:   путь к файлу, содержащему корпус материалов для обработки данных;

--save_tfidf: флаг для сохранения матрица TF-IDF в файл tfidf.dump;

--host_db:    адрес БД;

--user_db:    имя пользователя БД;

--pass_db:    пароль пользователя;

--db_name:    имя БД;
