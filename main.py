from view import user_interface as ui


def main():
    interface = ui.UserInterface()
    interface.run()
    interface.stop()


if __name__ == '__main__':
    main()
