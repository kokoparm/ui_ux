import os

KORS = tuple("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
ENGS = (
    "r",
    "R",
    "rt",
    "s",
    "sw",
    "sg",
    "e",
    "f",
    "fr",
    "fa",
    "fq",
    "ft",
    "fx",
    "fv",
    "fg",
    "a",
    "q",
    "qt",
    "t",
    "T",
    "d",
    "w",
    "c",
    "z",
    "x",
    "v",
    "g",
    "k",
    "o",
    "i",
    "O",
    "j",
    "p",
    "u",
    "P",
    "h",
    "hk",
    "ho",
    "hl",
    "y",
    "n",
    "nj",
    "np",
    "nl",
    "b",
    "m",
    "ml",
    "l",
)

eng_kor = dict(zip(ENGS, KORS))

# print(eng_kor)


def eng_to_kor(base_dir):
    folders = os.listdir(base_dir)

    for folder in folders:
        try:
            src = os.path.join(base_dir, folder)
            dst = os.path.join(base_dir, eng_kor[folder])
            os.rename(src, dst)
            print(src + " to " + dst)
        except KeyError as e:
            print(e, "is KOR")


if __name__ == "__main__":
    # base_dir = "./dataset/captures"
    base_dir = "./dataset/keypoints"
    eng_to_kor(base_dir)
