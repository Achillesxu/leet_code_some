--
-- Created by IntelliJ IDEA.
-- User: xushiyin
-- Date: 2017/12/13
-- Time: 20:48
-- To change this template use File | Settings | File Templates.
--
print('hello world')

local l_csjon = require ("cjson")
local sock = require ("socket")
local lfs = require ("lfs")

local json_data = '{"name":"tom", "age":"10"}'

local unjson = l_csjon.decode(json_data)

print(unjson["name"])
print(#unjson['age'])


a = {} -- create a table and store its reference in 'a'
k = "x"
a[k] = 10 -- new entry, with key="x" and value=10
a[20] = "great" -- new entry, with key=20 and value="great"
print(a["x"]) --> 10
k = 25
print(a[k]) --> "great"
a["x"] = a["x"] + 1 -- increments entry "x"
print(a["x"]) --> 11

local tolerance = 10
function isturnback (angle)
    angle = angle % 360
    return (math.abs(angle - 180) < tolerance)
end
print(isturnback(-180))

print(type(1 .. 2))

a = {10, 20, 30, nil, nil }
print(#a)
b = {10, nil, 30, nil, nil }
print(#b)
print(b[3])

function add_a(x)
    local sum = 0
    for i=1, #x do
        sum = sum + x[i]
    end
    return sum
end

names = {"Peter", "Paul", "Mary"}
grades = {Mary = 10, Paul = 7, Peter = 8 }

print(grades[names[1]])

function values (t)
local i = 0
return function () i = i + 1; return t[i] end
end

t = {10, 20, 30}
iter = values(t) -- creates the iterator
while true do
local element = iter() -- calls the iterator
if element == nil then break end
print(element)
end

c = {"one", "two", "three"}

for i, v in ipairs(c) do
print(i, v)
end

co = coroutine.create(function () print("hi") end)

print(co)

print(coroutine.status(co))

coroutine.resume(co)

print(coroutine.status(co))

List = {}
function List.new ()
return {first = 0, last = -1}
end

function List.pushfirst (list, value)
local first = list.first - 1
list.first = first
list[first] = value
end

function List.pushlast (list, value)
local last = list.last + 1
list.last = last
list[last] = value
end

function List.popfirst (list)
local first = list.first
if first > list.last then error("list is empty") end
local value = list[first]
list[first] = nil -- to allow garbage collection
list.first = first + 1
return value
end

function List.poplast (list)
local last = list.last
if list.first > last then error("list is empty") end
local value = list[last]
list[last] = nil -- to allow garbage collection
list.last = last - 1
return value
end

m_queue = List.new()
List.pushfirst (m_queue, 10)
List.pushlast (m_queue, 15)

for i, v in pairs(m_queue) do
    print(i, v)
end








