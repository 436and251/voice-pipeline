class StageGraph:
    def __init__(self, dependencies: dict[str, set[str]]):
        self.dependencies = {name: set(deps) for name, deps in dependencies.items()}
        unknown = {
            dependency
            for stage_dependencies in self.dependencies.values()
            for dependency in stage_dependencies
            if dependency not in self.dependencies
        }
        if unknown:
            raise ValueError(f"unknown stage dependencies: {', '.join(sorted(unknown))}")
        self.topological_order()

    def topological_order(self, target: str | None = None) -> list[str]:
        if target is not None and target not in self.dependencies:
            raise ValueError(f"unknown target stage: {target}")

        included = set(self.dependencies)
        if target is not None:
            included = {target}
            pending = [target]
            while pending:
                stage = pending.pop()
                for dependency in self.dependencies[stage]:
                    if dependency not in included:
                        included.add(dependency)
                        pending.append(dependency)

        remaining = {name: self.dependencies[name] & included for name in included}
        order: list[str] = []
        while remaining:
            ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
            if not ready:
                raise ValueError(f"stage dependency cycle: {', '.join(sorted(remaining))}")
            for name in ready:
                order.append(name)
                del remaining[name]
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return order

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
