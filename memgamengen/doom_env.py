"""ViZDoom environment wrapper for data collection and evaluation."""

import os
import numpy as np
import vizdoom as vzd
from typing import Optional, Tuple, Dict, List


# Action space: 8 discrete actions for DOOM
# [ATTACK, USE, MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
ACTION_NAMES = [
    "ATTACK", "USE", "MOVE_LEFT", "MOVE_RIGHT",
    "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"
]

# Single-action buttons (one action at a time for simplicity)
SINGLE_ACTIONS = [
    [1, 0, 0, 0, 0, 0, 0, 0],  # ATTACK
    [0, 1, 0, 0, 0, 0, 0, 0],  # USE
    [0, 0, 1, 0, 0, 0, 0, 0],  # MOVE_LEFT
    [0, 0, 0, 1, 0, 0, 0, 0],  # MOVE_RIGHT
    [0, 0, 0, 0, 1, 0, 0, 0],  # MOVE_FORWARD
    [0, 0, 0, 0, 0, 1, 0, 0],  # MOVE_BACKWARD
    [0, 0, 0, 0, 0, 0, 1, 0],  # TURN_LEFT
    [0, 0, 0, 0, 0, 0, 0, 1],  # TURN_RIGHT
]


def get_scenario_path(scenario: str) -> str:
    """Get the path to a ViZDoom scenario config file."""
    vizdoom_path = os.path.dirname(vzd.__file__)
    scenarios_dir = os.path.join(vizdoom_path, "scenarios")

    scenario_map = {
        "basic": "basic.cfg",
        "deadly_corridor": "deadly_corridor.cfg",
        "defend_the_center": "defend_the_center.cfg",
        "defend_the_line": "defend_the_line.cfg",
        "health_gathering": "health_gathering.cfg",
        "my_way_home": "my_way_home.cfg",
        "predict_position": "predict_position.cfg",
        "deathmatch": "deathmatch.cfg",
    }

    cfg_name = scenario_map.get(scenario, f"{scenario}.cfg")
    path = os.path.join(scenarios_dir, cfg_name)
    if not os.path.exists(path):
        # Fallback to basic
        path = os.path.join(scenarios_dir, "basic.cfg")
    return path


class DoomEnvironment:
    """Wrapper around ViZDoom for data collection and evaluation."""

    def __init__(
        self,
        scenario: str = "basic",
        resolution: Tuple[int, int] = (160, 120),
        frame_skip: int = 4,
        visible: bool = False,
        game_variables: Optional[List[str]] = None,
    ):
        self.scenario = scenario
        self.resolution = resolution
        self.frame_skip = frame_skip
        self.visible = visible
        self.game_variable_names = game_variables or ["HEALTH", "AMMO2"]

        self.game = vzd.DoomGame()
        self._setup_game()

        self.num_actions = len(SINGLE_ACTIONS)
        self.action_list = SINGLE_ACTIONS

    def _setup_game(self):
        """Configure the ViZDoom game instance."""
        config_path = get_scenario_path(self.scenario)
        self.game.load_config(config_path)

        # Screen settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        # Window mode
        self.game.set_window_visible(self.visible)
        self.game.set_render_hud(True)

        # Available buttons
        self.game.clear_available_buttons()
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.USE)
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)
        self.game.add_available_button(vzd.Button.TURN_LEFT)
        self.game.add_available_button(vzd.Button.TURN_RIGHT)

        # Game variables to track
        self.game.clear_available_game_variables()
        var_map = {
            "HEALTH": vzd.GameVariable.HEALTH,
            "AMMO2": vzd.GameVariable.AMMO2,
            "ARMOR": vzd.GameVariable.ARMOR,
            "KILLCOUNT": vzd.GameVariable.KILLCOUNT,
            "AMMO0": vzd.GameVariable.AMMO0,
            "AMMO1": vzd.GameVariable.AMMO1,
            "AMMO3": vzd.GameVariable.AMMO3,
        }
        for var_name in self.game_variable_names:
            if var_name in var_map:
                self.game.add_available_game_variable(var_map[var_name])

        # Misc settings
        self.game.set_episode_timeout(2100)  # ~35 seconds at 60fps
        self.game.set_living_reward(0)

        self.game.init()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return initial observation."""
        self.game.new_episode()
        state = self.game.get_state()
        obs = state.screen_buffer  # (H, W, 3) or (3, H, W) depending on format
        if obs.shape[0] == 3:
            obs = np.transpose(obs, (1, 2, 0))  # -> (H, W, 3)

        info = self._get_info(state)
        return obs, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        action = self.action_list[action_idx]
        reward = self.game.make_action(action, self.frame_skip)
        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros((120, 160, 3), dtype=np.uint8)
            info = {"game_variables": {name: 0.0 for name in self.game_variable_names}}
        else:
            state = self.game.get_state()
            obs = state.screen_buffer
            if obs.shape[0] == 3:
                obs = np.transpose(obs, (1, 2, 0))
            info = self._get_info(state)

        return obs, reward, done, info

    def _get_info(self, state) -> Dict:
        """Extract game variables from state."""
        info = {"game_variables": {}}
        if state is not None and state.game_variables is not None:
            for i, name in enumerate(self.game_variable_names):
                if i < len(state.game_variables):
                    info["game_variables"][name] = float(state.game_variables[i])
        return info

    def close(self):
        """Close the environment."""
        self.game.close()

    def get_available_actions(self) -> int:
        return self.num_actions


class DoomEnvironmentDefendCenter(DoomEnvironment):
    """Defend the center scenario - good for longer episodes with combat."""

    def __init__(self, resolution=(160, 120), frame_skip=4, visible=False):
        super().__init__(
            scenario="defend_the_center",
            resolution=resolution,
            frame_skip=frame_skip,
            visible=visible,
            game_variables=["HEALTH", "AMMO2", "KILLCOUNT"],
        )


class DoomEnvironmentDeadlyCorridor(DoomEnvironment):
    """Deadly corridor scenario - good for navigation + combat."""

    def __init__(self, resolution=(160, 120), frame_skip=4, visible=False):
        super().__init__(
            scenario="deadly_corridor",
            resolution=resolution,
            frame_skip=frame_skip,
            visible=visible,
            game_variables=["HEALTH", "AMMO2", "ARMOR"],
        )
