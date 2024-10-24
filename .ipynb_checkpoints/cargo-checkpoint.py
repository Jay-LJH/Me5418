
class cargo:
    
    def __init__(self, env, x, y, width, height, expire_time):
        self.env = env
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.expire_time = expire_time
        self.expired = False
        self.carry = False
        self.env.box_matrix[x][y] = expire_time
        self.collision_enter = False

    def step(self, timedelta):
        pass

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        pass

    '''
    def reward(self,car):
        reward = 0  
        distance = euclidean((car.hull.position), (self.x, self.y))
        if distance < custom_parameter.crash_distance: # close enough to pick up the box
            if (math.sqrt(car.hull.linearVelocity.lengthSquared) < custom_parameter.crash_speed \
                and car.carry is None):   # pick up the box
                car.carry = self
                self.carry = True
                reward += custom_parameter.pickup_reward
                self.env.box_matrix[self.x][self.y] = 0
                self.destroy_on_map() # destroy the box on the map
                logger.log("Pick up")

            elif self.collision_enter == False: # if crash for the first time
                reward += custom_parameter.crash_reward
                self.collision_enter = True
                logger.log("Crash")
        else:
            self.collision_enter = False

        if(self.env.t > self.expire_time): # if the box is expired
            if self.expired:
                reward += custom_parameter.expire_reward_continuous
            else:
                self.expired = True
                reward += custom_parameter.expire_reward
                logger.log("Expire")

        if self.env.t>self.expire_time + custom_parameter.destory_time: #destroy the box after it expired for 20 seconds
            self.destroy_on_map()
            self.destroy()
            if car.carry == self:
                car.carry = None
            logger.log("Destroy")
        return reward

    def destroy_on_map(self):
        self.env.box_matrix[self.x][self.y] = 0
        # self.env.world.DestroyBody(self.body)
        
    def destroy(self):
        self.env.box_list.remove(self)
    '''