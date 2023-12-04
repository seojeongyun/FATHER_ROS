#!/usr/bin/env python3

import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((200,200))

def getKey(keyName):
    pygame.event.clear()
    result = False
    for events in pygame.event.get(): pass
    KeyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if KeyInput[myKey]:
        result = True
    pygame.display.update()
    return result
