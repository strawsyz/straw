import pygame
from plane_sprites import *




class PlaneGame:
    def __init__(self, width=None, height=None):
        # 这个初始化，貌似可有可无？
        pygame.init()
        # self.WIDTH = width
        self.win = pygame.display.set_mode(WIN_RECT.size)
        # 创建时钟对象
        self.clock = pygame.time.Clock()
        # 创建精灵组
        self.__create_sprites()
        # 设置定时器事件
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 1000)
        pygame.time.set_timer(HERO_FIIR_EVNET, 200)
        print('init')

    def start_game(self):
        print('game start ')
        while True:
            self.clock.tick(FRAME_PER_SEC)
            self.__event_handler()
            self.__check_collide()
            self.__update_sprite()
            pygame.display.update()

    def __event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                PlaneGame.game_over()
            elif event.type == CREATE_ENEMY_EVENT:
                enemy = Enemy()
                self.enemy_group.add(enemy)
                print('enemy on stage')
            # elif event.type == pygame.KEYDOWN and event.key == pygame.BUTTON_RIGHT:
            #     print('go to right once')
            elif event.type == HERO_FIIR_EVNET:
                self.hero.fire()
                print('fire event ')

        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_RIGHT]:
            self.hero.speed = 3
        elif keys_pressed[pygame.K_LEFT]:
            self.hero.speed = -3
        else:
            self.hero.speed = 0

    def __check_collide(self):
        pygame.sprite.groupcollide(self.hero.bullets_group, self.enemy_group, True, True)

        enemies = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)
        if len(enemies) > 0:
            self.hero.kill()
            PlaneGame.game_over()
        pass

    def __update_sprite(self):
        self.back_group.update()
        self.back_group.draw(self.win)
        self.enemy_group.update()
        self.enemy_group.draw(self.win)
        self.hero_group.update()
        self.hero_group.draw(self.win)
        self.hero.bullets_group.update()
        self.hero.bullets_group.draw(self.win)


    @staticmethod
    def game_over():
        print('game over')
        pygame.quit()
        exit()

    def __create_sprites(self):
        print('create sprites')
        bg1 = Background()
        bg2 = Background(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)

        self.enemy_group = pygame.sprite.Group()

        self.hero = Hero()
        self.hero_group = pygame.sprite.Group(self.hero)


if __name__ == '__main__':
    game = PlaneGame()
    game.start_game()
