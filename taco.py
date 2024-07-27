from enum import Enum
from pydantic import BaseModel

class SpiceLevel(str, Enum):
    mild = "Mild"
    medium = "Medium"
    hot = "Hot"


class SugarLevel(str, Enum):
    light = "Light"
    regular = "Regular"
    extra = "Extra"


class CrunchyTaco(BaseModel):
    extra_cheese: bool = False
    no_lettuce: bool = False
    add_sour_cream: bool = False
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild


class SoftTaco(BaseModel):
    grilled_chicken: bool = False
    extra_tomatoes: bool = False
    add_guacamole: bool = False


class DoritosLocosTaco(BaseModel):
    supreme: bool = False
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild
    extra_nacho_cheese: bool = False


class CheesyGorditaCrunch(BaseModel):
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild
    jalapenos: bool = False
    extra_cheese: bool = False


class BeanBurrito(BaseModel):
    add_rice: bool = False
    grilled: bool = False
    extra_cheese: bool = False
    add_sour_cream: bool = False


class Beefy5LayerBurrito(BaseModel):
    no_beans: bool = False
    extra_beef: bool = False
    add_rice: bool = False
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild


class BurritoSupreme(BaseModel):
    grilled_chicken: bool = False
    extra_guacamole: bool = False
    no_onions: bool = False


class Quesarito(BaseModel):
    steak: bool = False
    extra_chipotle_sauce: SpiceLevel = SpiceLevel.mild
    add_jalapenos: bool = False


class CrunchwrapSupreme(BaseModel):
    black_beans: bool = False
    add_jalapenos: bool = False
    extra_nacho_cheese: bool = False


class MexicanPizza(BaseModel):
    add_chicken: bool = False
    extra_tomatoes: bool = False
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild


class ChalupaSupreme(BaseModel):
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild
    extra_cheese: bool = False
    add_guacamole: bool = False


class Quesadilla(BaseModel):
    steak: bool = False
    extra_cheese: bool = False
    add_jalapenos: bool = False


class NachosBellGrande(BaseModel):
    add_jalapenos: bool = False
    extra_beef: bool = False
    no_beans: bool = False
    add_guacamole: bool = False


class ChipsAndNachoCheeseSauce(BaseModel):
    add_jalapenos: bool = False
    upgrade_to_large: bool = False
    cheese_spiciness: SpiceLevel = SpiceLevel.mild


class CheesyFiestaPotatoes(BaseModel):
    extra_cheese: bool = False
    add_bacon_bits: bool = False
    spicy_ranch_sauce: SpiceLevel = SpiceLevel.mild


class BlackBeans(BaseModel):
    add_rice: bool = False
    cheese_on_top: bool = False


class CinnamonTwists(BaseModel):
    extra_cinnamon_sugar: SugarLevel = SugarLevel.light


class CinnabonDelights(BaseModel):
    upgrade_to_4_pack: bool = False
    extra_icing: SugarLevel = SugarLevel.light


class BajaBlastFreeze(BaseModel):
    add_strawberry_syrup: bool = False
    upgrade_to_large: bool = False


class Pepsi(BaseModel):
    no_ice: bool = False
    upgrade_to_large: bool = False


class CheeseBeanAndRiceBurrito(BaseModel):
    grilled: bool = False
    add_jalapenos: bool = False
    extra_cheese: bool = False


class SpicyPotatoSoftTaco(BaseModel):
    add_beef: bool = False
    extra_cheese: bool = False
    spicy_sauce: SpiceLevel = SpiceLevel.mild


class BeefyMeltBurrito(BaseModel):
    add_rice: bool = False
    grilled: bool = False
    extra_cheese_sauce: bool = False


class Order(BaseModel):
    crunchy_taco: CrunchyTaco
    soft_taco: SoftTaco
    doritos_locos_taco: DoritosLocosTaco
    cheesy_gordita_crunch: CheesyGorditaCrunch
    bean_burrito: BeanBurrito
    beefy_5_layer_burrito: Beefy5LayerBurrito
    burrito_supreme: BurritoSupreme
    quesarito: Quesarito
    crunchwrap_supreme: CrunchwrapSupreme
    mexican_pizza: MexicanPizza
    chalupa_supreme: ChalupaSupreme
    quesadilla: Quesadilla
    nachos_bell_grande: NachosBellGrande
    chips_and_nacho_cheese_sauce: ChipsAndNachoCheeseSauce
    cheesy_fiesta_potatoes: CheesyFiestaPotatoes
    black_beans: BlackBeans
    cinnamon_twists: CinnamonTwists
    cinnabon_delights: CinnabonDelights
    baja_blast_freeze: BajaBlastFreeze
    pepsi: Pepsi
    cheese_bean_and_rice_burrito: CheeseBeanAndRiceBurrito
    spicy_potato_soft_taco: SpicyPotatoSoftTaco
    beefy_melt_burrito: BeefyMeltBurrito


JSON_SCHEMAS = {
    "crunchy_taco": CrunchyTaco.model_json_schema(),
    "soft_taco": SoftTaco.model_json_schema(),
    "doritos_locos_taco": DoritosLocosTaco.model_json_schema(),
    "cheesy_gordita_crunch": CheesyGorditaCrunch.model_json_schema(),
    "bean_burrito": BeanBurrito.model_json_schema(),
    "beefy_5_layer_burrito": Beefy5LayerBurrito.model_json_schema(),
    "burrito_supreme": BurritoSupreme.model_json_schema(),
    "quesarito": Quesarito.model_json_schema(),
    "crunchwrap_supreme": CrunchwrapSupreme.model_json_schema(),
    "mexican_pizza": MexicanPizza.model_json_schema(),
    "chalupa_supreme": ChalupaSupreme.model_json_schema(),
    "quesadilla": Quesadilla.model_json_schema(),
    "nachos_bell_grande": NachosBellGrande.model_json_schema(),
    "chips_and_nacho_cheese_sauce": ChipsAndNachoCheeseSauce.model_json_schema(),
    "cheesy_fiesta_potatoes": CheesyFiestaPotatoes.model_json_schema(),
    "black_beans": BlackBeans.model_json_schema(),
    "cinnamon_twists": CinnamonTwists.model_json_schema(),
    "cinnabon_delights": CinnabonDelights.model_json_schema(),
    "baja_blast_freeze": BajaBlastFreeze.model_json_schema(),
    "pepsi": Pepsi.model_json_schema(),
    "cheese_bean_and_rice_burrito": CheeseBeanAndRiceBurrito.model_json_schema(),
    "spicy_potato_soft_taco": SpicyPotatoSoftTaco.model_json_schema(),
    "beefy_melt_burrito": BeefyMeltBurrito.model_json_schema(),
}