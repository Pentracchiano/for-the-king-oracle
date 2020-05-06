from typing import *
from scipy.stats import binom


class Character:

    def __init__(self, health: int, armor: int, resistance: int, is_friendly: bool = False):
        self._is_friendly = is_friendly
        self._health = health
        self._armor = armor
        self._resistance = resistance

    @property
    def armor(self) -> int:
        return self._armor

    @property
    def resistance(self) -> int:
        return self._resistance

    @property
    def is_friendly(self) -> bool:
        return self._is_friendly

    @property
    def health(self) -> int:
        return self._health

    @property
    def is_dead(self) -> bool:
        return self.health > 0

    def damage(self, attack: "Attack", tokens: int):
        self._health -= attack.get_true_damage(self, tokens)


class Attack:

    def __init__(self, total_damage: int, probability_per_token: float, max_tokens: int, is_physical: bool,
                 is_splash: bool = False):
        self._probability_per_token = probability_per_token
        self._max_tokens = max_tokens
        self._is_physical = is_physical
        self._damage_per_token = total_damage / max_tokens
        self._is_splash = is_splash

    @property
    def is_physical(self) -> bool:
        return self._is_physical

    @property
    def damage_per_token(self) -> float:
        return self._damage_per_token

    @property
    def is_splash(self) -> bool:
        return self._is_splash

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def probability_per_token(self) -> float:
        return self._probability_per_token

    def tokens_to_kill(self, character: Character) -> int:
        """
        Returns the minimum tokens needed to kill this character with this attack.
        It can be impossible to achieve in one turn.
        """
        true_health = self.get_character_true_health(character)
        min_tokens_to_kill = round(true_health / self.damage_per_token)
        return min_tokens_to_kill

    def can_kill_in_one_turn(self, character: Character) -> bool:
        return self.max_tokens <= self.tokens_to_kill(character)

    def get_true_damage(self, character: Character, tokens: int) -> int:
        reduction = self.get_character_reduction(character)
        damage = (self.damage_per_token * tokens) - reduction
        return round(damage) if damage > 0 else 0

    def get_character_reduction(self, character: Character) -> int:
        return character.armor if self.is_physical else character.resistance

    def get_character_true_health(self, character: Character) -> int:
        return character.health + self.get_character_reduction(character)


class Encounter:
    MAX_ENEMIES = 3

    def __init__(self, enemies: List[Character]):
        """

        :param enemies: a list of enemies, ordered with the first enemy on the left as 0th enemy in the list. It
                        will be padded with None elements if it contains less than three enemies.
        """
        self._enemies = enemies
        # Add padding to the enemies list
        total_enemies = len(self._enemies)
        if total_enemies != 3:
            self._enemies += [None] * (Encounter.MAX_ENEMIES - total_enemies)

    @property
    def enemies(self):
        return self._enemies

    def get_damage_exact_probabilities(self, attack: Attack, enemy_position: int) -> List[Mapping[str, float]]:
        # this method is here in order to account for splash damages in the future
        enemy = self.enemies[enemy_position]
        for i in range(attack.max_tokens + 1):
            damage = attack.get_true_damage(enemy, i)
            probability = binom.pmf(i, attack.max_tokens, attack.probability_per_token)
            yield {"damage": damage, "probability": probability}

    def get_damage_cumulative_probabilities(self, attack: Attack, enemy_position: int) -> List[Mapping[str, float]]:
        enemy = self.enemies[enemy_position]
        for i in range(attack.max_tokens + 1):
            damage = attack.get_true_damage(enemy, i)
            probability = binom.sf(i - 1, attack.max_tokens, attack.probability_per_token)
            yield {"damage": damage, "probability": probability}

    def get_kill_probability(self, attack: Attack, enemy_position: int) -> float:
        probability_to_kill = 0
        enemy = self.enemies[enemy_position]
        enemy_total_health = attack.get_character_true_health(enemy)

        for damage_probability in self.get_damage_exact_probabilities(attack, enemy_position):
            if damage_probability["damage"] >= enemy_total_health:
                probability_to_kill += damage_probability["probability"]

        return probability_to_kill

    def expected_damage(self, attack: Attack, enemy_position: int) -> float:
        expected_damage = 0
        for damage_probability in self.get_damage_exact_probabilities(attack, enemy_position):
            expected_damage += damage_probability["probability"] * damage_probability["damage"]

        return expected_damage

    def get_fail_probability(self, attack: Attack, enemy_position: int) -> float:
        for damage_probability in self.get_damage_exact_probabilities(attack, enemy_position):
            return damage_probability["probability"]


if __name__ == "__main__":
    enemy1 = Character(42, 3, 2)
    enemy2 = Character(10, 1, 0)
    attack = Attack(21, 0.78, 4, True)
    encounter = Encounter([enemy1, enemy2])

    p1 = list(encounter.get_damage_cumulative_probabilities(attack, 0))
    p2 = list(encounter.get_damage_cumulative_probabilities(attack, 1))

    testp1 = 0
    for item in p1:
        testp1 += item["probability"]

    testp2 = 0
    for item in p2:
        testp2 += item["probability"]

    print(testp1, testp2)

    print("\n\n")

    p1 = list(encounter.get_damage_exact_probabilities(attack, 0))
    p2 = list(encounter.get_damage_exact_probabilities(attack, 1))

    testp1 = 0
    for item in p1:
        testp1 += item["probability"]

    testp2 = 0
    for item in p2:
        testp2 += item["probability"]

    print(p1)
    print(p2)
    print(testp1, testp2)

    print("\n")

    print(encounter.expected_damage(attack, 0))
    print(encounter.expected_damage(attack, 1))

    print(encounter.get_kill_probability(attack, 1))
    print(encounter.get_fail_probability(attack, 1))
