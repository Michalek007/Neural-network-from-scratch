from cli.groups.seed import seed_cli
from cli.groups.seed.seed_group import SeedGroup


@seed_cli.command('init')
def init():
    """ Initialises database table with necessary data. """
    return SeedGroup().init()


@seed_cli.command('users')
def users():
    """ Seeds users table with tests data. """
    return SeedGroup().users()


@seed_cli.command('params')
def params():
    """ Seeds performance table with tests data. """
    return SeedGroup().params()


@seed_cli.command('digit_images')
def digit_images():
    """ Seeds digit_images table with tests data. """
    return SeedGroup().digit_images()
