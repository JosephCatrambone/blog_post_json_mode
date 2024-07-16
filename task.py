# Describes our assorted tasks:

import json
from enum import Enum
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Union


#
# Data Structures:
#

# Flat Named Entity Extraction
# Single Document -> Single JSON Object:
class NERMultiExtraction(BaseModel):
    names: list[str]
    locations: list[str]
    organizations: list[str]
    misc: list[str]


# Nested Named Entity Extraction
# Single Document -> Many JSON Entries with ENUM constraint:
class _NERObject(str, Enum):
    name = 'name'
    location = 'location'
    organization = 'organization'
    misc = 'misc'


class _NERExtraction(BaseModel):
    text: str
    extraction_type: _NERObject


class NERNestedExtraction(BaseModel):
    named_entities: list[_NERExtraction]


# "Thing" extraction.
# Similar to NER, but without a hard restriction on a type.
class ThingExtraction(BaseModel):
    text: str
    thingtype: str


class NestedThingExtraction(BaseModel):
    things: list[ThingExtraction]


# Unit Extraction
class UnitExtraction(BaseModel):
    quantity: float
    unit: str


class NestedUnitExtraction(BaseModel):
    items: list[UnitExtraction]


# Event Extraction
class GeographicLocation(BaseModel):
    latitude: float
    longitude: float


class NamedLocation(BaseModel):
    fine: str = Field(description="The finest/narrowest defined region, for example: Kyiv, San Francisco, Wabash and Lake, Navy Pier.")
    coarse: str = Field(description="A coarse disambiguating region, for example: California, the Midwest, France.")


class EventExtraction(BaseModel):
    year: int
    month: int = Field(le=12)  # Zero indexed?  One indexed?
    day: int = Field(le=31)
    hour: int = Field(le=23)
    minute: int = Field(le=60)
    name: str
    location: Optional[Union[GeographicLocation, NamedLocation]]


class NestedEventExtraction(BaseModel):
    events: list[EventExtraction]


#
# Descriptors and Sample Data:
#

class Task(str, Enum):
    NER_FLAT = "ner_flat"
    NER_NESTED = "ner_nested"
    THING_EXTRACTION = "thing_extraction"
    UNIT_EXTRACTION = "unit_extraction"
    EVENT_EXTRACTION = "event_extraction"


DATA_MODELS = {
    Task.NER_FLAT: NERMultiExtraction,
    Task.NER_NESTED: NERNestedExtraction,
    Task.THING_EXTRACTION: NestedThingExtraction,
    Task.UNIT_EXTRACTION: NestedUnitExtraction,
    Task.EVENT_EXTRACTION: NestedEventExtraction,
}


SCHEMAS = {
    t: json.dumps(v.model_json_schema()) for t, v in DATA_MODELS.items()
    #Task.NER_FLAT: json.dumps(NERMultiExtraction.model_json_schema()),
    #Task.NER_NESTED: json.dumps(NERNestedExtraction.model_json_schema()),
    #Task.THING_EXTRACTION: json.dumps(NestedThingExtraction.model_json_schema()),
    #Task.UNIT_EXTRACTION: json.dumps(NestedUnitExtraction.model_json_schema()),
    #Task.EVENT_EXTRACTION: json.dumps(NestedEventExtraction.model_json_schema()),
}


# NuExtract largely fails with the JSON Schema standard. We need to custom-define the schema for the examples.
NUEXTRACT_SCHEMAS = {
    Task.NER_FLAT:
"""{
    "names": [],
    "organizations": [],
    "locations": [],
    "misc": []
}""",
    Task.NER_NESTED:
"""{
    "named_entities": [
        {
            "text": "",
            "extraction_type": "name|location|organization|misc"
        }
    ]
}""",
    Task.THING_EXTRACTION:
"""{
    "things": [
        {
            "text": "",
            "thingtype": ""
        }
    ]
}""",
    Task.UNIT_EXTRACTION:
"""{
    "items": [
        {
            "quantity": "",
            "unit": ""
        }
    ]
}""",
    Task.EVENT_EXTRACTION:
"""{
    "events": [
        {
            "year": "",
            "month": "",
            "day": "",
            "hour": "",
            "minute": "",
            "name": "",
            "location": {
                "fine": "",
                "coarse": "",
                "latitude": "",
                "longitude": ""
            }
        }
    ]
}"""
}


EXAMPLES = {
    Task.NER_FLAT: [
"""{
    "names": ["Bob Loblaw", "Orson Wells", "Aaron Spacemuseum"],
    "organizations": ["Bob Loblaw's Law Blog", "Air and Space Museum"],
    "locations": ["Washington D.C."],
    "misc": []
}""",
"""{
    "names": [],
    "organizations": ["NASA", "The National Aeronotics and Space Administration"],
    "locations": [],
    "misc": ["Straight Outta Compton"]
}""",
"""{
    "names": [],
    "organizations": [],
    "locations": ["San Francisco", "Oakland, CA"],
    "misc": []
}"""
    ],
    Task.NER_NESTED: [
"""{
    "named_entities": [
        {
            "text": "Aaron Spacemuseum",
            "extraction_type": "name"
        },
        {
            "text": "NASA",
            "extraction_type": "organization"
        },
        {
            "text": "San Francisco",
            "extraction_type": "location"
        }
    ]
}""",
"""{
    "named_entities": [
        {
            "text": "Norm Alman",
            "extraction_type": "name"
        },
        {
            "text": "Dark Side of the Moon",
            "extraction_type": "misc"
        }
    ]
}""",
"""{
    "named_entities": [
        {
            "text": "Moe Thegrass",
            "extraction_type": "name"
        },
        {
            "text": "Huge Yakman",
            "extraction_type": "name"
        }
    ]
}""",],
    Task.THING_EXTRACTION: [
"""{
    "things": [
        {
            "text": "Pink Floyd",
            "thingtype": "band"
        },
        {
            "text": "San Francisco Bay Area",
            "thingtype": "location"
        },
        {
            "text": "google.com",
            "thingtype": "website"
        }
    ]
}""",
"""{
    "things": [
        {
            "text": "Riverside",
            "thingtype": "location"
        },
        {
            "text": "Bob Loblaw's Law Blog",
            "thingtype": "website"
        }
    ]
}""",
"""{
    "things": [
        {
            "text": "Earth",
            "thingtype": "planet"
        },
        {
            "text": "The Odyssey",
            "thingtype": "book"
        }
    ]
}"""],
    Task.UNIT_EXTRACTION:[
"""{
    "items": [
        {
            "quantity": "10",
            "unit": "oz"
        },
        {
            "quantity": "320",
            "unit": "ml"
        }
    ]
}""",
"""{
    "items": [
        {
            "quantity": "2.61",
            "unit": "lightyears"
        }
    ]
}""",
    """{
    "items": [
        {
            "quantity": "3.0x10^8",
            "unit": "m/s"
        },
        {
            "quantity": "9.81",
            "unit": "m/s^2"
        }
    ]
}""",],
    Task.EVENT_EXTRACTION: [
"""{
    "events": [
        {
            "year": 2024,
            "month": 7,
            "day": 13,
            "hour": null,
            "minute": null,
            "name": "Fusion Breakthrough",
            "location": {
                "fine": "Lawrence Livermore Labs",
                "coarse": "California"
            }
        }
    ]
}""",
"""{
    "events": [
        {
            "year": 2023,
            "month": 12,
            "day": 28,
            "hour": 12,
            "minute": 0,
            "name": "Largest snowfall ever recorded",
            "location": {
                "fine": "Minneapolis",
                "coarse": "Minnesota"
            }
        },
        {
            "year": 2023,
            "month": 12,
            "day": 29,
            "hour": 18,
            "minute": 0,
            "name": "Snow plow deployment paused",
            "location": {
                "latitude": 44.986656,
                "longitude": -93.258133
            }
        }
    ]
}""",
"""{
    "events": [
        {
            "year": 1999,
            "month": 1,
            "day": 1,
            "hour": 0,
            "minute": 1,
            "name": "New Year's Day Celebration",
            "location": {
                "fine": "Times Square",
                "coarse": "United States"
            }
        }
    ]
}""",
    ]}

