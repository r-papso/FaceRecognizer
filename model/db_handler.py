import sqlite3
import numpy as np
from model.identity_model import Identity


class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def get_instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `get_instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


@Singleton
class DBHandler:
    def __init__(self):
        self.connection = sqlite3.connect('resources/identitydb')
        self.identities = self.load_identities()

    def __del__(self):
        self.connection.close()

    def get_identities(self):
        return self.identities

    def add_identity(self, identity):
        str_features = ','.join(map(str, identity.features))
        cursor = self.connection.cursor()
        cursor.execute('INSERT INTO identities (name, surname, feature_vector) VALUES (?,?,?)',
                       (identity.name, identity.surname, str_features))
        self.connection.commit()
        identity.identity_id = cursor.lastrowid
        self.identities[identity.identity_id] = identity

    def edit_identity(self, identity):
        cursor = self.connection.cursor()
        if identity.features is not None:
            str_features = ','.join(map(str, identity.features))
            sql = 'UPDATE identities ' \
                  'SET name = ?, surname = ?, feature_vector = ? ' \
                  'WHERE identity_id = ?'
            cursor.execute(sql, (identity.name, identity.surname, str_features, identity.identity_id))
        else:
            sql = 'UPDATE identities ' \
                  'SET name = ?, surname = ? ' \
                  'WHERE identity_id = ?'
            cursor.execute(sql, (identity.name, identity.surname, identity.identity_id))
            identity.features = self.identities[identity.identity_id].features
        self.connection.commit()
        self.identities[identity.identity_id] = identity

    def delete_identity(self, identity_id):
        sql = 'DELETE FROM identities WHERE identity_id = ?'
        cur = self.connection.cursor()
        cur.execute(sql, (identity_id,))
        self.connection.commit()
        del self.identities[identity_id]

    def load_identities(self):
        cursor = self.connection.cursor()
        identities = dict()
        for row in cursor.execute('SELECT * FROM identities'):
            features = np.fromstring(row[3], dtype=np.float32, sep=',')
            identities[row[0]] = Identity(name=row[1], surname=row[2], identity_id=row[0], features=features)
        return identities
