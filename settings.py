import typing


class Settings:
    """
    Settings class, uses a JSON file
    """

    # Wall tokens
    wallTokens: typing.List[str]

    # Start tokens
    startTokens: typing.List[str]

    # Hole tokens
    holeTokens: typing.List[str]

    # Wall thickness
    thickness: int

    # Display zoom
    zoom: int

    # Euler time
    eulerTime: float

    # Delta t
    dt: float

    # Simulation ball speed
    ballSpeed: float

    # Animate heat progression or not
    animate: bool

    # Use Dirichlet or Neumann
    dirichlet: bool

    # Verbose or quiet
    verbose: bool

    def __init__(self,
                 wallTokens: typing.List[str],
                 startTokens: typing.List[str],
                 holeTokens: typing.List[str],
                 thickness: int,
                 zoom: int,
                 eulerTime: float,
                 dt: float,
                 ballSpeed: float,
                 animate: bool,
                 dirichlet: bool,
                 verbose: bool):
        self.wallTokens = wallTokens
        self.startTokens = startTokens
        self.holeTokens = holeTokens
        self.thickness = thickness
        self.zoom = zoom
        self.eulerTime = eulerTime
        self.dt = dt
        self.ballSpeed = ballSpeed
        self.animate = animate
        self.dirichlet = dirichlet
        self.verbose = verbose
