import platform

__system = platform.system()
if __system == "Windows":
    image_path = "D:\Download\datasets\polyp\\05\data"
    mask_path = "D:\Download\datasets\polyp\\05\mask"

    pretrain_path = "D:\Download\models\polyp\FCN_NLL_ep1989_04-57-10.pkl"
    model_save_path = "D:\Download\models\polyp"
    result_save_path = "D:\Download\models\polyp\\result"

elif __system == "Linux":
    image_path = "/home/straw/Downloads/dataset/polyp/TMP/06/data"
    mask_path = "/home/straw/Downloads/dataset/polyp/TMP/06/mask"

    pretrain_path = ""
    model_save_path = "/home/straw/Downloads/models/polyp/"
    result_save_path = "/home/straw/Download\models\polyp\\result"
