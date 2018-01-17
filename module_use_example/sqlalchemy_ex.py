#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/26 上午9:47
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : sqlalchemy_ex.py
@desc :
"""
from datetime import datetime
from sqlalchemy import create_engine, ForeignKey, Boolean
from sqlalchemy import Table, Column, Integer, Numeric, String, DateTime
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker


m_engine = create_engine('mysql+pymysql://root:Filter1986@localhost/alchemy', echo=True, pool_recycle=3600)

m_conn = m_engine.connect()

Base = declarative_base()


class Cookie(Base):
    """
    table cookie definition
    """
    __tablename__ = 'cookies'
    cookie_id = Column(Integer(), primary_key=True, autoincrement=True)
    cookie_name = Column(String(50), index=True)
    cookie_recipe_url = Column(String(255))
    cookie_sku = Column(String(60))
    quantity = Column(Integer())
    unit_cost = Column(Numeric(12, 2))

    def __repr__(self):
        return '{name}(cookie_name={self.cookie_name}, cookie_recipe_url={self.cookie_recipe_url' \
               '}, cookie_sku={self.cookie_sku}, quantity={self.quantity}, unit_cost={self.unit_cost})'\
            .format(name=type(self).__name__, self=self)


class LineItem(Base):
    """
    table lineitem definition
    """
    __tablename__ = 'line_items'
    line_item_id = Column(Integer(), primary_key=True)
    order_id = Column(Integer(), ForeignKey('orders.order_id'))
    cookie_id = Column(Integer(), ForeignKey('cookies.cookie_id'))
    quantity = Column(Integer())
    extended_cost = Column(Numeric(12, 2))

    order = relationship('Order', backref=backref('line_items', order_by=line_item_id))
    cookie = relationship('Cookie', uselist=False)

    def __repr__(self):
        return '{name}(order_id={self.order_id}, cookie_id={self.cookie_id}, ' \
               'quantity={self.quantity}, extended_cost={self.extended_cost})'.\
            format(name=type(self).__name__, self=self)


class User(Base):
    """
    table users definition
    """
    __tablename__ = 'users'
    user_id = Column(Integer(), primary_key=True)
    user_name = Column(String(64), nullable=False, unique=True)
    email_address = Column(String(255), nullable=False)
    phone_number = Column(String(16), nullable=False, unique=True)
    password = Column(String(64), nullable=False)
    created_on = Column(DateTime(), default=datetime.now)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return '{name}(user_name={self.user_name}, email_address={self.email_address}, ' \
               'phone_number={self.phone_number}, password={self.password})'.format(name=type(self).__name__, self=self)


class Order(Base):
    """
    table orders definition
    """
    __tablename__ = 'orders'
    order_id = Column(Integer(), primary_key=True)
    user_id = Column(Integer(), ForeignKey('users.user_id'))
    shipped = Column(Boolean(), default=False)

    user = relationship("User", backref=backref('orders', order_by=order_id))

    def __repr__(self):
        return '{name}(user_id={self.user_id}, shipped={self.shipped})'.format(name=type(self).__name__, self=self)


Base.metadata.create_all(m_engine)

Session = sessionmaker(bind=m_engine)

m_session = Session()


def insert_cookie(cookie_name=None, cookie_recipe_url=None, cookie_sku=None, quantity=None, unit_cost=None):
    ins_cookie = Cookie(cookie_name=cookie_name, cookie_recipe_url=cookie_recipe_url,
                        cookie_sku=cookie_sku, quantity=quantity, unit_cost=unit_cost)
    m_session.add(ins_cookie)
    m_session.commit()


def insert_cookie_many():
    c1 = Cookie(cookie_name='peanut butter',
                cookie_recipe_url='http://some.aweso.me/cookie/peanut.html',
                cookie_sku='PB01',
                quantity=24,
                unit_cost=0.25)
    c2 = Cookie(cookie_name='oatmeal raisin',
                cookie_recipe_url='http://some.okay.me/cookie/raisin.html',
                cookie_sku='EWW01',
                quantity=100,
                unit_cost=1.00)
    m_session.bulk_save_objects([c1, c2])
    m_session.commit()


if __name__ == '__main__':
    # insert_cookie(cookie_name='chocolate chip',
    #               cookie_recipe_url='http://some.aweso.me/cookie/recipe.html',
    #               cookie_sku='CC01',
    #               quantity=12,
    #               unit_cost=0.05)
    # insert_cookie_many()

    # query data in databases
    # for cookie in m_session.query(Cookie):
    #     print(cookie)
    # print(m_session.query(Cookie.cookie_name, Cookie.quantity).first())
    # for cookie in m_session.query(Cookie).order_by(Cookie.quantity):
    #     print('{:3} - {}'.format(cookie.quantity, cookie.cookie_name))
    # inv_count = m_session.query(func.sum(Cookie.quantity)).scalar()
    # print(inv_count)

    # update databases
    # query = m_session.query(Cookie)
    # cc_cookie = query.filter(Cookie.cookie_name == "chocolate chip").first()
    # cc_cookie.quantity = cc_cookie.quantity + 120
    # m_session.commit()
    # print(cc_cookie.quantity)
    # query = m_session.query(Cookie)
    # query = query.filter(Cookie.cookie_name == "chocolate chip")
    # query.update({Cookie.quantity: Cookie.quantity - 20})
    # cc_cookie = query.first()
    # print(cc_cookie.quantity)

    # cookiemon = User(user_name='cookiemon',
    #                  email_address='mon@cookie.com',
    #                  phone_number='111-111-1111',
    #                  password='password')
    # cakeeater = User(user_name='cakeeater',
    #                  email_address='cakeeater@cake.com',
    #                  phone_number='222-222-2222',
    #                  password='password')
    # pieperson = User(user_name='pieperson',
    #                  email_address='person@pie.com',
    #                  phone_number='333-333-3333',
    #                  password='password')
    # m_session.add(cookiemon)
    # m_session.add(cakeeater)
    # m_session.add(pieperson)
    # m_session.commit()
    # cookiemon = m_session.query(User).filter(User.user_name == 'cookiemon').one()
    # o1 = Order()
    # o1.user = cookiemon
    # m_session.add(o1)
    #
    # cc = m_session.query(Cookie).filter(Cookie.cookie_name == "chocolate chip").one()
    # line1 = LineItem(cookie=cc, quantity=2, extended_cost=1.00)
    # pb = m_session.query(Cookie).filter(Cookie.cookie_name == "peanut butter").one()
    # line2 = LineItem(quantity=12, extended_cost=3.00)
    # line2.cookie = pb
    # line2.order = o1
    #
    # o1.line_items.append(line1)
    # o1.line_items.append(line2)
    # m_session.commit()

    # cakeeater = m_session.query(User).filter(User.user_name == 'cakeeater').one()
    # o2 = Order()
    # o2.user = cakeeater
    # cc = m_session.query(Cookie).filter(Cookie.cookie_name == "chocolate chip").one()
    # line1 = LineItem(cookie=cc, quantity=24, extended_cost=12.00)
    # oat = m_session.query(Cookie).filter(Cookie.cookie_name == "oatmeal raisin").one()
    # line2 = LineItem(cookie=oat, quantity=6, extended_cost=6.00)
    # o2.line_items.append(line1)
    # o2.line_items.append(line2)
    # m_session.add(o2)
    # m_session.commit()

    query = m_session.query(Order.order_id, User.user_name, User.phone_number,
                            Cookie.cookie_name, LineItem.quantity,
                            LineItem.extended_cost)
    query = query.join(User).join(LineItem).join(Cookie)
    for result in query.filter(User.user_name == 'cookiemon'):
        print(result)

    query = m_session.query(User.user_name, func.count(Order.order_id))
    query = query.outerjoin(Order).group_by(User.user_name)
    for row in query:
        print(row)

    print('over')
