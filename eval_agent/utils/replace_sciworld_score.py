from scienceworld import ScienceWorldEnv


def step(self, inputStr:str):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))        # Convert from 0-1 to 0-100
    task_completed = bool(
        self.server.getCompleted()
    )
    isCompleted = task_completed
    numMoves = self.getNumMoves()

    # Calculate reward
    reward = score - self.lastStepScore         # Calculate reward (delta score) for this step
    self.lastStepScore = score                  # Store current score for reward calculation on the next step

    terminated_by_step_limit = (
        numMoves > self.envStepLimit
    )
    terminated_by_negative_score = score < 0

    if terminated_by_step_limit:
        isCompleted = True

    if terminated_by_negative_score:
        isCompleted = True

    # Mirror of Jericho API
    infos = {
        'moves': numMoves,
        'raw_score': raw_score,
        'score': score,
        'reward': reward,
        'look': self.look(),
        'inv': self.inventory(),
        'taskDesc': self.taskdescription(),
        'valid': self.getValidActionObjectCombinations(),
        'variationIdx': self.variationIdx,
        'taskName': self.taskName,
        'simplificationStr': self.simplificationStr,
        "task_completed": task_completed,
        "terminated_by_step_limit": (
            terminated_by_step_limit
        ),
        "terminated_by_negative_score": (
            terminated_by_negative_score
        ),
    }

    return observation, reward, isCompleted, infos


def sciworld_monkey_patch():
    ScienceWorldEnv.step = step
    print("Monkey Patched ScienceWorldEnv.step")
