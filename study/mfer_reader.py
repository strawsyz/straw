paths = [
    "D:\(lab\graduate\work\mfer file study\ECG01.mwf",
    "D:\(lab\graduate\work\mfer file study/ECG04.mwf",
    "D:\(lab\graduate\work\mfer file study/ECG05.mwf"
]
path = paths[0]
with open(path, "rb") as f:
    # res = chardet.detect(f.read())
    print(f.read())
    # print()
print(16 ** 6)
# MFER allows designation of an infinite data length by encoding 0x80 for the data length. This infinite length
# designation is terminated by encoding the end-of-contents (tag = 00, data length = 00). In MFER, the infinite length
# designation is available only with the tag number 0x3F (definition of the channel attribute).
# This indicates that the data section is blank. In MFER, this resets concerned items in the root definition to default
# values and places the channel definition in the condition that the root definition designates.

# The header or waveform data values are encoded in the value section according to descriptors specified by the tag.

# mfer中的所有值都有默认值
# 频道信息 P/C=1 tag number是1F

# 厂商：NIHON KOHDEN 机种：QP901D机种版本00-80型号0001
#     b'@ MFR Standard 12 leads ECG       \x17\x1eNIHON KOHDEN^QP901D^00-80^0001\x
# 厂商：Suzuken Co. Ltd. 机种：Cardico1208 机种版本1.00
# b'@  MFER Standard 12 Leads ECG     \x17!Suzuken Co. Ltd.^Cardico1208^1.00\x01\
# b'@ MFR Standard 12 leads ECG       \x17\x0cGE Marquette\x0
