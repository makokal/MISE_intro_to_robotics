
from navaid import NavAid


if __name__ == '__main__':
    navi = NavAid()

    # navi.prepare_to_calibrate(30)
    # navi.calibrate()

    navi.run(full_operation_mode=True)

    navi.send_instructions(None)