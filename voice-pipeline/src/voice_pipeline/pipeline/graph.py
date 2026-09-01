class StageGraph:
    def __init__(self, dependencies: dict[str, set[str]]):
        self.dependencies = {name: set(deps) for name, deps in dependencies.items()}

    def downstream_of(self, stage: str) -> set[str]:
        found: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name, deps in self.dependencies.items():
                if name in found or name == stage:
                    continue
                if stage in deps or deps & found:
                    found.add(name)
                    changed = True
        return found
