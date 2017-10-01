DROP TABLE mass_media_newsfeed.posts;
DROP TABLE mass_media_newsfeed.comments;
DROP TABLE mass_media_newsfeed.mass_media;

CREATE TABLE mass_media_newsfeed.mass_media (
  id          INTEGER NOT NULL UNIQUE PRIMARY KEY,
  name        TEXT    NOT NULL,
  domain      TEXT    NOT NULL,
  description TEXT,
  activity    TEXT
);

CREATE TABLE mass_media_newsfeed.posts (
  id_mass_media  INTEGER NOT NULL REFERENCES mass_media_newsfeed.mass_media (id),
  id             INTEGER NOT NULL UNIQUE PRIMARY KEY,
  date           INTEGER NOT NULL,
  post_type      TEXT    NOT NULL,
  short_text     TEXT    NOT NULL,
  text           TEXT,
  link           TEXT,
  count_comments INTEGER NOT NULL,
  count_likes    INTEGER NOT NULL,
  count_reports  INTEGER NOT NULL
);

CREATE TABLE mass_media_newsfeed.comments (
  post_id     INTEGER NOT NULL,
  comment_id  INTEGER NOT NULL,
  date        INTEGER NOT NULL,
  text        TEXT    NOT NULL,
  count_likes INTEGER NOT NULL
);

CREATE TABLE mass_media_newsfeed.words_emotions (
  word                TEXT,
  negative            INTEGER,
  surprise            INTEGER,
  sadness             INTEGER,
  anger               INTEGER,
  disgust             INTEGER,
  contempt            INTEGER,
  grief_suffering     INTEGER,
  shame               INTEGER,
  interest_excitement INTEGER,
  guilt               INTEGER,
  confusion           INTEGER,
  gladness            INTEGER,
  counter             INTEGER
);


CREATE FUNCTION mass_media_newsfeed.add_word(word_                TEXT,
                                             negative_            INTEGER,
                                             surprise_            INTEGER,
                                             sadness_             INTEGER,
                                             anger_               INTEGER,
                                             disgust_             INTEGER,
                                             contempt_            INTEGER,
                                             grief_suffering_     INTEGER,
                                             shame_               INTEGER,
                                             interest_excitement_ INTEGER,
                                             guilt_               INTEGER,
                                             confusion_           INTEGER,
                                             gladness_            INTEGER)
  RETURNS INTEGER
LANGUAGE plpgsql
AS $$
BEGIN
  IF mass_media_newsfeed.except_word($1) = 0
  THEN
    INSERT INTO mass_media_newsfeed.words_emotions (word,
                                                    negative,
                                                    surprise,
                                                    sadness,
                                                    anger,
                                                    disgust,
                                                    contempt,
                                                    grief_suffering,
                                                    shame,
                                                    interest_excitement,
                                                    guilt,
                                                    confusion,
                                                    GLADNESS,
                                                    counter)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 1);
  ELSIF mass_media_newsfeed.except_word($1) = 1
    THEN
      UPDATE mass_media_newsfeed.words_emotions
      SET (negative,
           surprise,
           sadness,
           anger,
           disgust,
           contempt,
           grief_suffering,
           shame,
           interest_excitement,
           guilt,
           confusion,
           GLADNESS,
           counter) =
      (negative + $2,
        surprise + $3,
        sadness + $4,
        anger + $5,
        disgust + $6,
        contempt + $7,
        grief_suffering + $8,
        shame + $9,
        interest_excitement + $10,
        guilt + $11,
        confusion + $12,
       GLADNESS + $13,
       counter + 1)
      WHERE mass_media_newsfeed.words_emotions.word = $1;
  END IF;
  RETURN mass_media_newsfeed.except_word($1);
END;
$$;

CREATE FUNCTION mass_media_newsfeed.except_word(word_ TEXT)
  RETURNS TABLE(count INTEGER)
AS $$ SELECT count(word) :: INTEGER
      FROM mass_media_newsfeed.words_emotions
      WHERE words_emotions.word = $1;
$$
LANGUAGE SQL;
