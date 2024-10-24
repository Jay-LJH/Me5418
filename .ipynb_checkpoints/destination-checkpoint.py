class destination:
    
    def __init__(self, env, x, y, width, height, expire_time):
        self.env = env
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.expire_time = expire_time
        self.expired = False
        self.carry = False
        self.collision_enter = False

    def step(self, timedelta):
        pass

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        pass