import os

MODEL_PATH = "model/tic_tac_toe_cnn.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model missing. Put tic_tac_toe_cnn.h5 into /model folder.")


drive linkinden modeli indirdikten sonra kodun başına import ekleyip modeli çekmesi lazım

https://drive.google.com/file/d/1t2gsakNOWdH2QQt1FG4eXFV6NEtmyc4w/view?usp=sharing
