class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        if hour == 12:
            hour_angle = minutes / 2
        else:
            hour_angle = hour * 30 + minutes / 2
        minutes_angle = minutes * 6
        angle = abs(hour_angle - minutes_angle)
        return 360 - angle if angle > 180 else angle


if __name__ == '__main__':
    s = Solution()
    hour = 1
    minutes = 57
    res = s.angleClock(hour, minutes)
    print(res)
