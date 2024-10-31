"""
General housekeeping utilities.
Functions:
    arguments: handles command line arguments
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License

# Standard imports
import argparse
from sys import modules, exit as sys_exit

# Local application imports
import synth_utils

def arguments() -> dict:
    """
    Handle command line arguments.
    Returns:
        None
    """

    parser = argparse.ArgumentParser(description='Generate files of synthetic disk images.')
    parser.add_argument('--about',
                        help='Provide description, version, GNU license',
                        action='store_true',
                        default=False)
    parser.add_argument('--start_idx', '-s',
                        help='Starting index number of image files names (default: 0).',
                        metavar='IDX',
                        default=0,
                        type=int,
                        nargs='?',)
    parser.add_argument('--num_files', '-n',
                        help='Number of files to create (default: 25).',
                        default='25',
                        metavar='N',
                        type=int,
                        nargs='?',)

    args = parser.parse_args()

    about_text = (f'{modules["__main__"].__doc__}\n'
                  f'{"Author:".ljust(13)}{synth_utils.__author__}\n'
                  f'{"Version:".ljust(13)}{synth_utils.__version__}\n'
                  f'{"Status:".ljust(13)}{synth_utils.__status__}\n'
                  f'{"URL:".ljust(13)}{synth_utils.URL}\n'
                  f'{synth_utils.__copyright__}'
                  f'{synth_utils.__license__}\n'
                  )

    if args.about:
        print('====================== ABOUT START ====================')
        print(about_text)
        print('====================== ABOUT END ====================')
        sys_exit(0)

    arg_dict = {
        'about': about_text,
        'start_idx': args.start_idx,
        'num_files': args.num_files,
    }

    return arg_dict
