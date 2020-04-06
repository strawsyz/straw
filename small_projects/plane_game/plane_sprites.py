import random
import pygame

WIN_RECT = pygame.Rect(0, 0, 480, 700)
FRAME_PER_SEC = 60
CREATE_ENEMY_EVENT = pygame.USEREVENT
HERO_FIIR_EVNET = pygame.USEREVENT + 1


class GameSprite(pygame.sprite.Sprite):
    """飞机大战精灵"""

    def __init__(self, image_name, speed=1):
        # 调用父类的初始化方法
        super().__init__()
        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    def update(self):
        self.rect.y += self.speed


class Background(GameSprite):
    def __init__(self, is_alt=False):
        """is_alt,判断是否是第一张图"""
        super().__init__("images/background.png")
        if is_alt:
            self.rect.y = -self.rect.height

    def update(self):
        # 调用父类的方法实现
        super().update()
        if self.rect.y >= WIN_RECT.height:
            self.rect.y = -self.rect.height


class Enemy(GameSprite):
    """敌机"""

    def __init__(self):
        speed = random.randint(4, 8)
        super().__init__("images/enemy1.png", speed)
        self.rect.bottom = 0
        max_x = WIN_RECT.width - self.rect.width
        self.rect.x = random.randint(0, max_x)

    def update(self):
        super().update()
        if self.rect.y >= WIN_RECT.height:
            print('enemy gone')
            self.kill()

    def __del__(self):
        # print('enemy is dead', self.rect)
        pass


class Hero(GameSprite):
    """英雄精灵"""

    def __init__(self):
        super().__init__('images/me1.png', 0)
        self.rect.centerx = WIN_RECT.centerx
        self.rect.bottom = WIN_RECT.bottom - 120
        self.bullets_group = pygame.sprite.Group()

    def update(self):
        self.rect.x += self.speed
        if self.rect.x <= 0:
            self.rect.x = 0
        elif self.rect.right >= WIN_RECT.width:
            self.rect.right = WIN_RECT.width

    def fire(self):
        print('fire')
        bullet = Bullet()
        bullet.rect.bottom = self.rect.top
        bullet.rect.centerx = self.rect.centerx
        self.bullets_group.add(bullet)


class Bullet(GameSprite):
    def __init__(self):
        super().__init__('images/bullet1.png', -2)

    def update(self):
        super().update()
        if self.rect.bottom < 0:
            self.kill()

    def __del__(self):
        print('bullet dead')
