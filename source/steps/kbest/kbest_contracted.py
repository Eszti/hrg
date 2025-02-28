from source.steps.kbest.kbest import KBest


class KBestContracted(KBest):

    def __init__(self, config=None):
        super().__init__(
            description="Script to search k best contracted derivations in parsed charts.",
            script_name="kbest_contracted",
            config=config,
        )
        self.pos_tag_resolution = True
        self.gold_triplets_fn = "gold_contracted_triplets.json"


if __name__ == "__main__":
    KBestContracted().run()
