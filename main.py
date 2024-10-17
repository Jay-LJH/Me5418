if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])
    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CustomCarRacing()

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        s, r, terminated= env.step(a)
        while True:
            register_input()
            s, r, terminated= env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated :
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                positions = [(i, j) for i, row in enumerate(s["box"]) for j, value in enumerate(row) if value != 0]
                print(f"boxes: {positions}")
            steps += 1
            if terminated or restart or quit:
                break
    env.close()