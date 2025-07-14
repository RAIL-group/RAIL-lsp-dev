class Fluent:
    __slots__ = ("name", "args", "negated", "_hash")

    def __init__(self, name: str, *args: str, negated: bool = False):
        if args:
            self.name = name
            self.args = args
            if "not" == name:
                raise ValueError("Use the 'negated' argument or ~Fluent to negate.")
            self.negated = negated
        else:
            if negated:
                raise ValueError(
                    "Cannot both pass a full string and negated=True. Use 'not' or ~Fluent."
                )
            split = name.split(" ")
            if split[0] == "not":
                self.negated = True
                split = split[1:]
            else:
                self.negated = False
            self.name = split[0]
            self.args = tuple(split[1:])

        self._hash = hash((self.name, self.args, self.negated))

    def __str__(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.name} {' '.join(self.args)}"

    def __repr__(self) -> str:
        return f"F({self})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fluent) and self._hash == other._hash

    def __invert__(self) -> "Fluent":
        return Fluent(self.name, *self.args, negated=not self.negated)