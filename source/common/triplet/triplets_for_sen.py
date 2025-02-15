import json

from source.common.triplet.triplet import Triplet


class TripletsForSen:
    def __init__(self, triplets, sen_id, sen_text):
        super().__init__()
        self.sen_id = sen_id
        self.sen_text = sen_text
        self.triplets = sorted(triplets, key=lambda x: x.derivation_score, reverse=True)

    @staticmethod
    def from_json(fn):
        with open(fn) as f:
            triplets_for_sen_dict = json.load(f)
        return TripletsForSen(
            [
                Triplet(
                    t["labels"],
                    t["triplet_id"],
                    t.get("pred_resolution"),
                    t["derivation_score"],
                )
                for t in triplets_for_sen_dict["triplets"]
            ],
            triplets_for_sen_dict["sen_id"],
            triplets_for_sen_dict["sen_text"],
        )

    def to_json(self, fn):
        json_dict = dict()
        json_dict["sen_id"] = self.sen_id
        json_dict["sen_text"] = self.sen_text
        json_dict["triplets"] = [t.to_dict() for t in self.triplets]
        with open(fn, "w") as f:
            json.dump(json_dict, f, indent=4)

    def save_summary(self, fn):
        with open(fn, "w") as f:
            for triplet in self.triplets:
                f.write(f"{triplet.to_short_json()}\n")
