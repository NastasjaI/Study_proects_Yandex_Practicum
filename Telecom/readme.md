# Прогноз оттока клиентов сотового оператора
## Цель<br>
Обучить модель прогнозировать отток клиентов оператора связи, чтобы можно было создавать персональные предложени для клиентов, желающих прекратить сотрудничество.<br>
<br>
## Выводы<br>
В первую очередь был составлен предварительнвй план работы, для того, чтобы понимать возможные моментры реализации поставленной задачи и пруктурировать саму работу.<br>
1. Общие пункты плата были выполнены, однако по ходу работы план корректировался взависимотсти от обнаруженных данных и поведения модели:
- Пункт "Оценка и анализ данных" подпункт "Визуализировать данные, проверить зависимости признаков и (при необходимости) объединить таблицы" частично перенесён в следующий пункт, поскольку решила, что целесообразнее бутет проверять зависимость признаков после добавления целевого признака.
- Пункт "Подготовка данных для построения  моделей" выполнен полностью.
- Пункт "Моделирование" выполнен в полном объёме.
- Обоснование и выводы в работе присутствуют.
2. Задача была интересная, не очень сожная, но трудности всё-таки возникли:
- Выделение важных признаков, добавление новых и удаление менее значимых проводилось методом эксперемента с постоянным обучением модели и попыткой понять наличие утечки данных.
- Подобрать параметы для модели таким образом, чтобы она не переобучилась. а метрика качества была требуемого уровня.
3. Самые важные шаги, по моему мнению: подготовка, добавление и удаление признаков, поскольку каждый из них может существенно влиять на модель. Для лучшего понимания важности признаков применяла `phik`, так как практически все данные категориальные.
4. Все модели, которые были обучены, показали относительно хорощие результаты. Но в результате проведённой работы была выделена модель CatBoost, которая на увеличенной выборке (для нивелирования дисбаланса данных) показала качество AUC-ROC 0.993788 и 0.918285 на трениоровочной и тестовой данных соотвественно (график ниже).
Исходя из приведённого выще для решения текущей задачи, а именно - прогнозирования оттока клиентов Оператора связи «Ниединогоразрыва.ком» рекомендую использовать именно CatBoost на увеличенной выборке, учитывая подготовленный для её обучения набор признаков. <br>
<br>

## Проект завершён